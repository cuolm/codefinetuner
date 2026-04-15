
import pathlib
import textwrap
from pathlib import Path
 
import pytest
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.convert.config import Config
 
 
# --- Fixtures ---
 
 
@pytest.fixture
def config(mocker) -> Config:
    """
    Load a convert Config from the test YAML.
    _sync_converter_script_version is patched to prevent any network calls
    during construction, it is tested separately below.
    """
    mocker.patch.object(Config, "_sync_converter_script_version")
    return Config.load_from_yaml(test_config_path)
 
 
# --- Config.load_from_yaml ---
 
def test_load_from_yaml_success(config):
    assert config.workspace_path is not None
 
 
def test_load_from_yaml_missing_file(tmp_path, mocker):
    mocker.patch.object(Config, "_sync_converter_script_version")
    nonexistent_yaml = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        Config.load_from_yaml(nonexistent_yaml)
 
 
def test_load_from_yaml_invalid_yaml(tmp_path, mocker):
    mocker.patch.object(Config, "_sync_converter_script_version")
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("convert:\n  key: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load YAML"):
        Config.load_from_yaml(invalid_yaml)
 
 
def test_load_from_yaml_ignores_unknown_keys(tmp_path, mocker):
    """Extra YAML keys (e.g. from global anchors) must not raise."""
    mocker.patch.object(Config, "_sync_converter_script_version")
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


# --- _ensure_output_paths_exist ---
 
def test_ensure_output_paths_exist_creates_gguf_parent_dir(config):
    assert config.lora_model_gguf_path.parent.exists()


# --- _sync_converter_script_version ---

def test_sync_converter_script_version_skips_download_when_up_to_date(mocker, tmp_path):
    """No HTTP request is made when the script and version marker are both current."""
    workspace = tmp_path / "workspace"
    script_dir = workspace / "src" / "codefinetuner" / "convert"
    script_dir.mkdir(parents=True)
    
    script_path = script_dir / "convert_hf_to_gguf.py"
    script_path.write_text("# script", encoding="utf-8")
    
    version_marker = script_path.with_suffix(".version")
    version_marker.write_text("0.9.1")
    
    mocker.patch("importlib.metadata.version", return_value="0.9.1")
    http_mock = mocker.patch("codefinetuner.convert.config.httpx.Client")

    Config(workspace_path=workspace)

    http_mock.assert_not_called()
 
 
def test_sync_converter_script_version_downloads_when_old_version(mocker, tmp_path):
    """HTTP GET is issued when the local script file is outdated"""
    workspace = tmp_path / "workspace"
    script_dir = workspace / "src" / "codefinetuner" / "convert"
    script_dir.mkdir(parents=True)
    
    script_path = script_dir / "convert_hf_to_gguf.py"
    script_path.write_text("# old", encoding="utf-8")

    version_marker = script_path.with_suffix(".version")
    version_marker.write_text("0.5.4")

    mocker.patch("importlib.metadata.version", return_value="0.9.1")

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = "# downloaded script"
    mock_response.raise_for_status = mocker.Mock()
    mock_client = mocker.Mock()
    mock_client.get.return_value = mock_response
    mock_client.__enter__ = mocker.Mock(return_value=mock_client)
    mock_client.__exit__ = mocker.Mock(return_value=False)
    mocker.patch("codefinetuner.convert.config.httpx.Client", return_value=mock_client)

    test_config = Config(workspace_path=workspace)

    mock_client.get.assert_called_once()
    assert test_config.convert_hf_to_gguf_local_path.read_text(encoding="utf-8") == "# downloaded script"

 
