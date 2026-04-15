import pathlib
import textwrap

import pytest


tests_path = pathlib.Path(__file__).parent.parent 
test_data_path = tests_path / "data"
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.preprocess.config import Config


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- load_from_yaml ---

def test_load_from_yaml_success(config):
    assert config.model_name == "tests/models/Qwen2.5-Coder-0.5B"
    assert config.data_language == "c"
    assert ".c" in config.data_extensions


def test_load_from_yaml_missing_file(tmp_path):
    nonexistent_yaml = tmp_path / "nonexistent_yaml.yaml"
    with pytest.raises(FileNotFoundError):
        Config.load_from_yaml(nonexistent_yaml)


def test_load_from_yaml_invalid_yaml(tmp_path):
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("preprocess:\n  key: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse YAML"):
        Config.load_from_yaml(invalid_yaml)


def test_load_from_yaml_ignores_unknown_keys(tmp_path):
    """Extra YAML keys (e.g. from global anchors) must not raise."""
    config_text = textwrap.dedent("""
        preprocess:
          workspace_path: "tests"
          model_name: "Qwen/Qwen2.5-Coder-1.5B"
          fim_prefix_token: "<|fim_prefix|>"
          fim_middle_token: "<|fim_middle|>"
          fim_suffix_token: "<|fim_suffix|>"
          fim_pad_token: "<|fim_pad|>"
          eos_token: "<|endoftext|>"
          data_language: "c"
          data_extensions: [".c"]
          unknown_key_that_does_not_exist: 999
    """)
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    test_config = Config.load_from_yaml(test_config_path)
    assert test_config.model_name == "Qwen/Qwen2.5-Coder-1.5B"


# --- Ratio validation ---

def test_invalid_ratio_raises(tmp_path):
    config_text = textwrap.dedent("""
        preprocess:
          workspace_path: "tests"     
          model_name: "Qwen/Qwen2.5-Coder-1.5B"
          fim_prefix_token: "<|fim_prefix|>"
          fim_middle_token: "<|fim_middle|>"
          fim_suffix_token: "<|fim_suffix|>"
          fim_pad_token: "<|fim_pad|>"
          eos_token: "<|endoftext|>"
          data_language: "c"
          data_extensions: [".c"]
          train_ratio: 0.8
          eval_ratio: 0.3
          test_ratio: 0.1
    """)
    test_config_path = tmp_path / "bad_ratio.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    with pytest.raises(ValueError, match="ratios must sum to 1.0"):
        Config.load_from_yaml(test_config_path)


# --- _setup_paths ---

def test_setup_paths_dataset_paths_are_under_workspace(config):
    assert str(config.train_dataset_path).startswith(str(config.workspace_path))
    assert str(config.eval_dataset_path).startswith(str(config.workspace_path))
    assert str(config.test_dataset_path).startswith(str(config.workspace_path))


# --- _ensure_output_paths_exist ---

def test_ensure_output_paths_exist(config):
    assert config.train_dataset_path.parent.is_dir()
    assert config.eval_dataset_path.parent.is_dir()
    assert config.test_dataset_path.parent.is_dir()


# --- Language block loading ---

def test_block_types_loaded_as_sets(config):
    assert isinstance(config.tree_sitter_block_types, set)
    assert isinstance(config.tree_sitter_subblock_types, set)
    assert "function_definition" in config.tree_sitter_block_types
    assert "compound_statement" in config.tree_sitter_subblock_types


def test_unknown_language_raises(tmp_path):
    config_text = textwrap.dedent("""
        preprocess:
          workspace_path: "tests"                        
          model_name: "Qwen/Qwen2.5-Coder-1.5B"
          fim_prefix_token: "<|fim_prefix|>"
          fim_middle_token: "<|fim_middle|>"
          fim_suffix_token: "<|fim_suffix|>"
          fim_pad_token: "<|fim_pad|>"
          eos_token: "<|endoftext|>"
          data_language: "unknown"
          data_extensions: [".unknown"]
    """)
    test_config_path = tmp_path / "config.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        Config.load_from_yaml(test_config_path)


# --- Tree-sitter parser ---

def test_tree_sitter_parser_is_initialised(config):
    assert config.tree_sitter_parser is not None