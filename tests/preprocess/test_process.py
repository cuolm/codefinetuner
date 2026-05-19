import pathlib
import json
import textwrap
from transformers import AutoTokenizer

import pytest


tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.preprocess.config import Config
from codefinetuner.preprocess.process import (
    estimate_bytes_per_token_ratio,
    _extract_subblock_ranges,
    _filter_subblocks,
    _generate_fim_examples_from_code_block,
    create_fim_examples,
    _filter_tokenized_total_length,
    _filter_tokenized_middle_length,
    _save_tokenized_batch_as_jsonl,
    tokenize_filter_and_save 
)

# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load a Config from the test YAML, redirecting outputs to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()  # regenerates paths relative to the new workspace_path
    return test_config

@pytest.fixture
def tokenizer(config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.eos_token = config.eos_token
    return tokenizer

@pytest.fixture
def test_code_block(config):
    """
    Returns a specific, hardcoded code block (bytes and AST node) derived from 
    a known example snippet, bypassing the entire data loading pipeline for speed/determinism.
    """
    # Define the known good source code *inside* the fixture scope
    source_code_utf8 = textwrap.dedent("""
        int calculate_sum(int n) {
            if (n <= 0) {
                return 0;
            }
            return n + calculate_sum(n - 1);
        }
    """).encode('utf8')

    tree = config.tree_sitter_parser.parse(source_code_utf8)
    root_node = tree.root_node

    return (source_code_utf8, root_node)


# --- estimate_bytes_per_token_ratio ---

def test_estimate_bytes_per_token_ratio(config, tokenizer):
    bytes_per_token_ratio = estimate_bytes_per_token_ratio(config, tokenizer, 10)
    assert bytes_per_token_ratio > 1
    assert bytes_per_token_ratio < 5


# --- _extract_subblock_ranges ---
 
def test_extract_subblock_ranges_returns_list(config, test_code_block):
    code_bytes, node = test_code_block 
    ranges = _extract_subblock_ranges(config, node, base_offset=node.start_byte)
    assert isinstance(ranges, list)
 
 
def test_extract_subblock_ranges_offsets_are_relative(config, test_code_block):
    code_bytes, node = test_code_block
    ranges = _extract_subblock_ranges(config, node, base_offset=node.start_byte)
    for start, end in ranges:
        assert start >= 0
        assert end <= len(code_bytes)
        assert start < end


# --- _filter_subblocks ---
 
def test_filter_subblocks_keeps_ranges_within_threshold():
    ranges = [(0, 10), (0, 50), (0, 200)]
    result = _filter_subblocks(ranges, max_bytes_per_subblock=100)
    for _, end in result:
        assert end <= 100
 
 
def test_filter_subblocks_inclusive_boundary():
    ranges = [(0, 100)]
    result = _filter_subblocks(ranges, max_bytes_per_subblock=100)
    assert len(result) == 1
 
 
def test_filter_subblocks_returns_empty_when_all_exceed():
    result = _filter_subblocks([(0, 500)], max_bytes_per_subblock=100)
    assert result == []
    

# --- _generate_fim_examples_from_code_block ---
 
def test_generate_fim_examples_contains_all_special_tokens(config, test_code_block):
    code_bytes, _ = test_code_block
    ranges = [(17, 24), (25, 107), (31, 68), (43, 68), (53, 62), (73, 105)]
    bytes_per_token_ratio = 3

    examples = _generate_fim_examples_from_code_block(config, code_bytes, ranges, bytes_per_token_ratio=bytes_per_token_ratio)

    assert len(examples) > 0
    for example in examples:
        assert b"<|fim_prefix|>" in example
        assert b"<|fim_suffix|>" in example
        assert b"<|fim_middle|>" in example
        assert b"<|endoftext|>" in example


def test_generate_fim_examples_middle_tokens_length(config, test_code_block):
    code_bytes, _ = test_code_block
    ranges = [(17, 24), (25, 107), (31, 68), (43, 68), (53, 62), (73, 105)]
    bytes_per_token_ratio = 3
    min_num_of_middle_bytes = 30
    max_num_of_middle_bytes = 42
 
    config.min_middle_tokens_length = min_num_of_middle_bytes / bytes_per_token_ratio
    config.max_middle_tokens_length = max_num_of_middle_bytes / bytes_per_token_ratio
    examples = _generate_fim_examples_from_code_block(config, code_bytes, ranges, bytes_per_token_ratio=bytes_per_token_ratio)

    assert len(examples) > 0

    mid_token = config.fim_middle_token.encode()
    eos_token = config.eos_token.encode()
    for example in examples:
        parts = example.split(mid_token)
        after_middle = parts[1]
        middle_content = after_middle.split(eos_token)[0]
        assert len(middle_content) >= min_num_of_middle_bytes
        assert len(middle_content) <= max_num_of_middle_bytes

# --- create_fim_examples ---

def test_create_fim_examples(config, test_code_block):
    test_code_block_iter = iter([test_code_block])
    bytes_per_token_ratio = 3
    examples_iter = create_fim_examples(config, test_code_block_iter, bytes_per_token_ratio)
    assert hasattr(examples_iter, "__next__")
    examples = list(examples_iter)
    assert len(examples) > 0


# --- _filter_tokenized_total_length ---

def test_filter_tokenized_total_length_keeps_valid(config):
    config.max_token_sequence_length = 5
    batch = {
        "input_ids": [[1, 2, 3], [1, 2, 3, 4, 5]],
        "attention_mask": [[1, 1, 1], [1, 1, 1, 1, 1]],
        "labels": [[1, 2, 3], [1, 2, 3, 4, 5]]
    }
    
    result = _filter_tokenized_total_length(config, batch)
    
    assert len(result["input_ids"]) == 2
    assert result["input_ids"] == [[1, 2, 3], [1, 2, 3, 4, 5]]
    assert result["labels"] == [[1, 2, 3], [1, 2, 3, 4, 5]]


def test_filter_tokenized_total_length_removes_exceeding(config):
    config.max_token_sequence_length = 3
    batch = {
        "input_ids": [[1, 2], [1, 2, 3, 4]],
        "attention_mask": [[1, 1], [1, 1, 1, 1]],
        "labels": [[1, 2], [1, 2, 3, 4]]
    }
    
    result = _filter_tokenized_total_length(config, batch)
    
    assert len(result["input_ids"]) == 1
    assert result["input_ids"] == [[1, 2]]
    assert result["attention_mask"] == [[1, 1]]
    assert result["labels"] == [[1, 2]]


# --- _filter_tokenized_middle_length ---

def test_filter_tokenized_middle_length_keeps_within_bounds(config, tokenizer):
    config.min_middle_tokens_length = 2
    config.max_middle_tokens_length = 4
    mid_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    eos_id = tokenizer.convert_tokens_to_ids(config.eos_token)
    
    batch = {
        "input_ids": [[99, mid_id, 10, 11, eos_id]],
        "attention_mask": [[1, 1, 1, 1, 1]],
        "labels": [[99, mid_id, 10, 11, eos_id]]
    }
    
    result = _filter_tokenized_middle_length(config, tokenizer, batch)
    
    assert len(result["input_ids"]) == 1
    assert result["labels"] == [[99, mid_id, 10, 11, eos_id]]


def test_filter_tokenized_middle_length_removes_out_of_bounds(config, tokenizer):
    config.min_middle_tokens_length = 2
    config.max_middle_tokens_length = 3
    mid_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    eos_id = tokenizer.convert_tokens_to_ids(config.eos_token)
    
    batch = {
        "input_ids": [
            [99, mid_id, 10, eos_id],
            [99, mid_id, 10, 11, 12, 13, eos_id]
        ],
        "attention_mask": [
            [1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1]
        ],
        "labels": [
            [99, mid_id, 10, eos_id],
            [99, mid_id, 10, 11, 12, 13, eos_id]
        ]
    }
    
    result = _filter_tokenized_middle_length(config, tokenizer, batch)
    
    assert len(result["input_ids"]) == 0
    assert len(result["labels"]) == 0


def test_filter_tokenized_middle_length_drops_missing_token(config, tokenizer, caplog):
    config.min_middle_tokens_length = 1
    config.max_middle_tokens_length = 10
    
    batch = {
        "input_ids": [[1, 2, 3, 4]],
        "attention_mask": [[1, 1, 1, 1]],
        "labels": [[1, 2, 3, 4]]
    }
    
    result = _filter_tokenized_middle_length(config, tokenizer, batch)
    
    assert len(result["input_ids"]) == 0
    assert len(result["labels"]) == 0
    

# --- _save_tokenized_batch_as_jsonl ---
 
def test_save_tokenized_batch_writes_valid_jsonl(tmp_path):
    tmp_file_path = tmp_path / "tmp_file.jsonl"
    tokenized_batch = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
        "labels": [[1, 2, 3], [4, 5, 6]],
    }
    _save_tokenized_batch_as_jsonl(tmp_file_path, tokenized_batch)
 
    lines = tmp_file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    
    first_example = json.loads(lines[0])
    second_example = json.loads(lines[1])
    
    assert first_example["input_ids"] == [1, 2, 3]
    assert second_example["input_ids"] == [4, 5, 6]
    assert "attention_mask" in first_example
 

def test_save_tokenized_batch_appends_on_second_call(tmp_path):
    tmp_file_path = tmp_path / "tmp_file.jsonl"
    first_batch = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        "labels": [[1, 2, 3]],
    }
    second_batch = {
        "input_ids": [[4, 5, 6]],
        "attention_mask": [[1, 1, 0]],
        "labels": [[4, 5, 6]],
    }
    _save_tokenized_batch_as_jsonl(tmp_file_path, first_batch)
    _save_tokenized_batch_as_jsonl(tmp_file_path, second_batch)

    lines = tmp_file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first_batch_example = json.loads(lines[0])
    second_batch_example = json.loads(lines[1])
    assert first_batch_example["input_ids"] == [1, 2, 3]
    assert second_batch_example["input_ids"] == [4, 5, 6]


# --- tokenize_filter_and_save ---

def test_tokenize_filter_and_save(tmp_path, config, tokenizer):
    tmp_file_path = tmp_path / "tmp_file.jsonl"
    example = b'<|fim_prefix|>\nint calculate_sum(int n) {\n    <|fim_suffix|>}\n    return n + calculate_sum(n - 1);\n}\n<|fim_middle|> if (n <= 0) {\n        return 0;\n    <|endoftext|>'
    example_iter = iter([example])

    tokenize_filter_and_save(config, tmp_file_path, example_iter, tokenizer)

    assert tmp_file_path.exists()
    lines = tmp_file_path.read_text(encoding="utf-8").strip().splitlines()
    first_line_example = json.loads(lines[0])
    assert "input_ids" in first_line_example
    assert "attention_mask" in first_line_example
    assert "labels" in first_line_example
    