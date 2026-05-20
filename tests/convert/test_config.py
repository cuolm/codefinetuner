
import pathlib
import textwrap
from pathlib import Path
 
import pytest
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.convert.config import Config
 
 
# --- Fixtures ---
 
@pytest.fixture
def config(tmp_path) -> Config:
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config 
 
 
# --- load_from_yaml ---
 
def test_load_from_yaml_success(config):
    assert config.workspace_path is not None
 
 
def test_load_from_yaml_missing_file(tmp_path, mocker):
    nonexistent_yaml = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        Config.load_from_yaml(nonexistent_yaml)
 
 
def test_load_from_yaml_invalid_yaml(tmp_path, mocker):
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("convert:\n  key: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load YAML"):
        Config.load_from_yaml(invalid_yaml)
 
 
def test_load_from_yaml_ignores_unknown_keys(tmp_path, mocker):
    """Extra YAML keys (e.g. from global anchors) must not raise."""
    config_text = textwrap.dedent("""
        convert:
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
    assert test_config.workspace_path == Path("tests")


# --- _setup_paths ---
 
def test_setup_paths_under_workspace(config):
    assert str(config.lora_model_path).startswith(str(config.workspace_path))
    assert str(config.lora_model_gguf_path).startswith(str(config.workspace_path))
    assert str(config.convert_hf_to_gguf_local_path).startswith(str(config.workspace_path))
 
 
def test_setup_paths_lora_model_gguf_path_has_gguf_suffix(config):
    assert config.lora_model_gguf_path.suffix == ".gguf"

