import json
import pathlib

import pytest
from transformers import AutoTokenizer

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.evaluate.run import _ensure_output_paths_exist
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.benchmark import _extract_fim_parts, create_benchmark_dataset


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load an evaluate Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config 


@pytest.fixture
def tokenizer(config) -> AutoTokenizer:
    """Load the pinned local tokenizer from tests/models/."""
    return AutoTokenizer.from_pretrained(config.model_name)


@pytest.fixture
def example(config, tokenizer) -> list[int]:
    """
    Build a minimal but valid FIM token-id sequence from known text using the
    real tokenizer, so the expected prefix/suffix/middle strings are deterministic.
    Structure: <fim_prefix> prefix_text <fim_suffix> suffix_text <fim_middle> middle_text <eos>
    """
    fim_prefix_id = tokenizer.convert_tokens_to_ids(config.fim_prefix_token)
    fim_suffix_id = tokenizer.convert_tokens_to_ids(config.fim_suffix_token)
    fim_middle_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    eos_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    prefix_token_ids = tokenizer.encode("int n = 0;", add_special_tokens=False)
    suffix_token_ids = tokenizer.encode("return n;", add_special_tokens=False)
    middle_token_ids = tokenizer.encode("n += 1;", add_special_tokens=False)

    return (
        [fim_prefix_id]
        + prefix_token_ids
        + [fim_suffix_id]
        + suffix_token_ids
        + [fim_middle_id]
        + middle_token_ids
        + [eos_id]
    )


# --- _extract_fim_parts ---

def test_extract_fim_parts_success(config, tokenizer, example):
    result = _extract_fim_parts(config, tokenizer, example)

    prefix = "int n = 0;"
    suffix = "return n;"
    middle = "n += 1;"

    prefix_token_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    middle_token_ids = tokenizer.encode(middle, add_special_tokens=False)

    assert result["example_token_ids"] == example
    assert result["prefix_token_ids"] == prefix_token_ids  
    assert result["suffix_token_ids"] == suffix_token_ids 
    assert result["middle_token_ids"] == middle_token_ids 
    assert result["prefix"] == prefix
    assert result["suffix"] == suffix
    assert result["middle"] == middle


def test_extract_fim_parts_raises_error_if_token_missing(config, tokenizer, example):
    # missing the EOS token (last token)
    broken_sequence = example[:-1]
    
    with pytest.raises(ValueError, match="Missing FIM or EOS tokens in the sequence."):
        _extract_fim_parts(config, tokenizer, broken_sequence)


def test_extract_fim_parts_raises_error_if_wrong_order(config, tokenizer):
    fim_prefix_id = tokenizer.convert_tokens_to_ids(config.fim_prefix_token)
    fim_suffix_id = tokenizer.convert_tokens_to_ids(config.fim_suffix_token)
    fim_middle_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    eos_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    prefix_token_ids = tokenizer.encode("int n = 0;", add_special_tokens=False)
    suffix_token_ids = tokenizer.encode("return n;", add_special_tokens=False)
    middle_token_ids = tokenizer.encode("n += 1;", add_special_tokens=False)

    # Out of order sequence: Prefix -> Middle -> Suffix -> EOS
    scrambled_sequence = (
        [fim_prefix_id] + prefix_token_ids +
        [fim_middle_id] + middle_token_ids +
        [fim_suffix_id] + suffix_token_ids +
        [eos_id]
    )
    
    with pytest.raises(ValueError, match="FIM tokens are in the wrong order."):
        _extract_fim_parts(config, tokenizer, scrambled_sequence) 


# --- create_benchmark_dataset ---

def test_create_benchmark_dataset(config, tokenizer, example):
    _ensure_output_paths_exist(config)

    # Missing elements sequence to trigger pipeline skip logic
    invalid_sequence = [example[0], example[2]] 

    dummy_input_data = [
        {"input_ids": example, "attention_mask": [1] * len(example), "labels": example},
        {"input_ids": example, "attention_mask": [1] * len(example), "labels": example},
        {"input_ids": invalid_sequence, "attention_mask": [1, 1], "labels": invalid_sequence},
        {"input_ids": example, "attention_mask": [1] * len(example), "labels": example},
    ]
    
    config.test_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.test_dataset_path, "w", encoding="utf-8") as input_file:
        for item in dummy_input_data:
            input_file.write(json.dumps(item) + "\n")

    config.benchmark_sample_size = 2

    saved_count = create_benchmark_dataset(config)

    assert config.benchmark_dataset_path.exists()
    assert saved_count == 2
    
    output_lines_count = 0
    with open(config.benchmark_dataset_path, "r", encoding="utf-8") as output_file:
        for line in output_file:
            parsed_line = json.loads(line)
            assert "prefix" in parsed_line
            assert "suffix" in parsed_line
            assert "middle" in parsed_line
            output_lines_count += 1
            
    assert output_lines_count == 2