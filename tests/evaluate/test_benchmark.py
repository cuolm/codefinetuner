
import json
import pathlib
 
import pytest
from transformers import AutoTokenizer
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.benchmark import _extract_fim_parts, create_benchmark_dataset
 
 
# --- Fixtures ---
 
@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    return Config.load_from_yaml(test_config_path)
 
 
@pytest.fixture
def tokenizer(config) -> AutoTokenizer:
    """Load the pinned local tokenizer from tests/models/."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer


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
 
def test_extract_fim_parts(config, tokenizer, example):
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


# --- create_benchmark_dataset ---

def test_create_benchmark_dataset(config, tmp_path):
    """ uses tests/data/train/train_test.c """
    config.benchmark_dataset_path = tmp_path / "test_benchmark_dataset.jsonl"
    number_of_examples = 4
    config.benchmark_sample_size = number_of_examples 
    config.benchmark_min_fim_middle_tokens = 0

    return_count = create_benchmark_dataset(config)

    assert config.benchmark_dataset_path.exists()
    assert return_count == number_of_examples
    
    number_of_file_examples = 0
    with open(config.benchmark_dataset_path, 'r') as benchmark_dataset_file:
        for line in benchmark_dataset_file:
            number_of_file_examples+=1
    assert number_of_file_examples == number_of_examples
