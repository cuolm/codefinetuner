import pathlib
import textwrap

import pytest

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML and redirect paths to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- Config.load_from_yaml ---

def test_load_from_yaml_success(config):
    assert config.model_name == "tests/models/Qwen2.5-Coder-0.5B"
    assert config.fim_pad_token == "<|fim_pad|>"


def test_load_from_yaml_missing_file(tmp_path):
    nonexistent_yaml = tmp_path / "nonexistent_yaml.yaml"
    with pytest.raises(FileNotFoundError):
        Config.load_from_yaml(nonexistent_yaml)


def test_load_from_yaml_invalid_yaml(tmp_path):
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("finetune:\n  key: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load YAML"):
        Config.load_from_yaml(invalid_yaml)


def test_load_from_yaml_ignores_unknown_keys(tmp_path):
    """Extra YAML keys (e.g. from global anchors) must not raise."""
    config_text = textwrap.dedent("""
        finetune:
          workspace_path: "tests"
          model_name: "Qwen/Qwen2.5-Coder-1.5B"
          fim_prefix_token: "<|fim_prefix|>"
          fim_middle_token: "<|fim_middle|>"
          fim_suffix_token: "<|fim_suffix|>"
          fim_pad_token: "<|fim_pad|>"
          eos_token: "<|endoftext|>"
          label_pad_token_id: -100
          data_language: "c"
          data_extensions: [".c", ".h"]
          unknown_key_that_does_not_exist: 999
    """)
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    test_config = Config.load_from_yaml(test_config_path)
    assert test_config.model_name == "Qwen/Qwen2.5-Coder-1.5B"


# --- _setup_paths ---

def test_setup_paths_dataset_paths_are_under_workspace(config):
    assert str(config.train_dataset_path).startswith(str(config.workspace_path))
    assert str(config.eval_dataset_path).startswith(str(config.workspace_path))


# --- _ensure_output_paths_exist ---

def test_ensure_output_paths_exist_creates_parent_dirs(config):
    assert config.finetune_outputs_dir_path.parent.exists()
    assert config.trainer_checkpoints_dir_path.parent.exists()
    assert config.trainer_model_merge_offload_folder_path.parent.exists()
    assert config.trainer_log_path.parent.exists() 
    assert config.trainer_plot_path.parent.exists()
    assert config.lora_adapter_path.parent.exists()
    assert config.lora_model_path.parent.exists()


# --- _get_dataset_length ---

def test_get_dataset_length_counts_lines(config):
    length = config._get_dataset_length(config.train_dataset_path)
    assert length > 0


def test_get_dataset_length_raises_for_missing_file(config, tmp_path):
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        config._get_dataset_length(missing)


# --- _calculate_max_steps ---

def test_calculate_max_steps_returns_positive_value(config):
    assert config.trainer_max_steps > 0


def test_calculate_max_steps_returns_zero_for_empty_dataset(config):
    config.dataset_train_dataset_length = 0
    assert config._calculate_max_steps() == 0
