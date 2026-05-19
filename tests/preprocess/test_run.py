import pathlib
import json
import logging
from transformers import AutoTokenizer

import pytest


tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.preprocess.config import Config
from codefinetuner.preprocess.run import(
    _ensure_output_paths_exist,
    _clear_existing_datasets,
    _validate_and_configure_tokenizer,
    run
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


# --- test_ensure_output_paths_exist ---

def test_ensure_output_paths_exist(config):
    assert not config.preprocess_outputs_dir_path.exists()
    assert not config.train_dataset_path.parent.exists()
    
    _ensure_output_paths_exist(config)
    
    assert config.preprocess_outputs_dir_path.exists()
    assert config.preprocess_results_path.exists()
    assert config.train_dataset_path.parent.exists()


# --- _clear_existing_datasets ---

def test_clear_existing_datasets(config):
    _ensure_output_paths_exist(config)

    config.train_dataset_path.write_text("{}", encoding="utf-8")
    config.eval_dataset_path.write_text("{}", encoding="utf-8")
    config.test_dataset_path.write_text("{}", encoding="utf-8")
    
    _clear_existing_datasets(config)
    
    assert not config.train_dataset_path.exists()
    assert not config.eval_dataset_path.exists()
    assert not config.test_dataset_path.exists()


# --- _validate_and_configure_tokenizer ---

def test_validate_and_configure_tokenizer(config, tokenizer):
    _validate_and_configure_tokenizer(config, tokenizer)
    assert tokenizer.pad_token == config.fim_pad_token
    assert tokenizer.eos_token == config.eos_token
    

def test_validate_and_configure_tokenizer_raises_on_unknown_token(config, tokenizer):
    config.fim_prefix_token = "<|this_token_does_not_exist|>"
    with pytest.raises(ValueError, match="missing from Tokenizer"):
        _validate_and_configure_tokenizer(config, tokenizer)


# --- run ---

def test_run_integration_pipeline(config):
    run(config)
    
    assert config.train_dataset_path.exists()
    assert config.eval_dataset_path.exists()
    assert config.test_dataset_path.exists()
 
    for path in [config.train_dataset_path, config.eval_dataset_path, config.test_dataset_path]:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        
        assert len(lines) > 0
        
        record = json.loads(lines[0])
        assert "input_ids" in record
        assert "labels" in record
        assert record["input_ids"] == record["labels"]
