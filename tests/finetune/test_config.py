import pathlib
import textwrap

import pytest

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load a Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config.raw_data_path = None  # set to none, recalculate it in _setup_paths()
    test_config._setup_paths()  # regenerates paths relative to the new workspace_path
    return test_config


# --- load_from_yaml ---

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
          max_token_sequence_length: 1024
          data_language: "c"
          data_extensions: [".c", ".h"]
          use_unsloth: True
          unknown_key_that_does_not_exist: 999
    """)
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    test_config = Config.load_from_yaml(test_config_path)
    assert test_config.model_name == "Qwen/Qwen2.5-Coder-1.5B"


# --- _validate_config ---

def test_validate_config_best_strategy_matching_steps_ok(config):
    config.selected_checkpoint_strategy = "best"
    config.trainer_eval_steps = 25
    config.trainer_save_steps = 25
    config._validate_config()  # should not raise


def test_validate_config_best_strategy_mismatched_steps_raises(config):
    config.selected_checkpoint_strategy = "best"
    config.trainer_eval_steps = 25
    config.trainer_save_steps = 50
    with pytest.raises(ValueError, match="requires trainer_eval_steps"):
        config._validate_config()


def test_validate_config_last_strategy_mismatched_steps_ok(config):
    config.selected_checkpoint_strategy = "last"
    config.trainer_eval_steps = 25
    config.trainer_save_steps = 50
    config._validate_config()  


# --- _setup_paths ---

def test_setup_paths_all_paths_are_initialized_correctly(config):
    path_attributes = [
        "train_dataset_path",
        "eval_dataset_path",
        "finetune_outputs_dir_path",
        "trainer_checkpoints_dir_path",
        "trainer_model_merge_offload_folder_path",
        "trainer_log_path",
        "trainer_plot_path",
        "selected_checkpoint_path",
        "lora_model_path"
    ]
    
    for attr in path_attributes:
        path = getattr(config, attr)
        assert isinstance(path, pathlib.Path), f"{attr} should be a Path object"
        assert str(path).startswith(str(config.workspace_path)), f"{attr} is not under the workspace"
        assert path.is_absolute(), f"{attr} should be an absolute path"


# --- _ensure_output_paths_exist ---

def test_ensure_output_paths_exist_creates_parent_dirs(config):
    from codefinetuner.finetune.run import _ensure_output_paths_exist
    
    _ensure_output_paths_exist(config)
    
    paths_to_check = [
        "finetune_outputs_dir_path",
        "trainer_checkpoints_dir_path",
        "trainer_model_merge_offload_folder_path",
        "trainer_log_path",
        "trainer_plot_path",
        "selected_checkpoint_path",
        "lora_model_path"
    ]
    
    for attr in paths_to_check:
        path = getattr(config, attr)
        assert path.parent.exists(), f"Parent directory for {attr} ({path.parent}) was not created."
