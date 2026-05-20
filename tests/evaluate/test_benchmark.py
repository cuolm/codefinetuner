
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
def tokenizer(mocker):
    tokenizer_instance_mock = mocker.patch("transformers.AutoTokenizer")

    def dummy_convert_tokens_to_ids(token: str) -> int:
        token_to_id_map = {
            "<|fim_prefix|>": 1,
            "<|fim_suffix|>": 2,
            "<|fim_middle|>": 3,
            "<|endoftext|>": 4
        }
        return token_to_id_map.get(token, -1)

    def dummy_decode(token_ids: list[int], **kwargs) -> str:
        if token_ids == [10, 11]:
            return "int n = 0;"
        if token_ids == [20, 21]:
            return "return n;"
        if token_ids == [30, 31]:
            return "n += 1;"
        return ""

    tokenizer_instance_mock.convert_tokens_to_ids.side_effect = dummy_convert_tokens_to_ids
    tokenizer_instance_mock.decode.side_effect = dummy_decode

    return tokenizer_instance_mock 
 

# --- _extract_fim_parts ---
 
def test_extract_fim_parts_success(config, tokenizer):
    config.fim_prefix_token = "<|fim_prefix|>"
    config.fim_middle_token = "<|fim_middle|>"
    config.fim_suffix_token = "<|fim_suffix|>"
    config.fim_pad_token =  "<|fim_pad|>"
    config.eos_token = "<|endoftext|>"
    valid_token_sequence = [1, 10, 11, 2, 20, 21, 3, 30, 31, 4]
    result = _extract_fim_parts(config, tokenizer, valid_token_sequence)

    assert result["example_token_ids"] == valid_token_sequence
    assert result["prefix_token_ids"] == [10, 11]
    assert result["suffix_token_ids"] == [20, 21]
    assert result["middle_token_ids"] == [30, 31]
    assert result["prefix"] == "int n = 0;"
    assert result["suffix"] == "return n;"
    assert result["middle"] == "n += 1;"


def test_extract_fim_parts_raises_error_if_token_missing(config, tokenizer):
    # missing the EOS token (ID 4)
    broken_sequence = [1, 10, 11, 2, 20, 21, 3, 30, 31]
    
    with pytest.raises(ValueError, match="Missing FIM or EOS tokens in the sequence."):
        _extract_fim_parts(config, tokenizer, broken_sequence)


def test_extract_fim_parts_raises_error_if_wrong_order(config, tokenizer):
    # FIM markers out of order: Prefix (1), Middle (3), Suffix (2), EOS (4)
    scrambled_sequence = [1, 10, 11, 3, 30, 31, 2, 20, 21, 4]
    
    with pytest.raises(ValueError, match="FIM tokens are in the wrong order."):
        _extract_fim_parts(config, tokenizer, scrambled_sequence) 


# --- create_benchmark_dataset ---

def test_create_benchmark_dataset(config, tokenizer, mocker):
    _ensure_output_paths_exist(config)
    mocker.patch(
        "codefinetuner.evaluate.benchmark.AutoTokenizer.from_pretrained", 
        return_value=tokenizer
    )

    valid_token_sequence = [1, 10, 11, 2, 20, 21, 3, 30, 31, 4]
    dummy_input_data = [
        {"input_ids": valid_token_sequence, "attention_mask": [1] * 10, "labels": valid_token_sequence},
        {"input_ids": valid_token_sequence, "attention_mask": [1] * 10, "labels": valid_token_sequence},
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]}, # Invalid sequence to test skip logic
        {"input_ids": valid_token_sequence, "attention_mask": [1] * 10, "labels": valid_token_sequence},
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
