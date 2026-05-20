import pathlib

import pytest

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.convert.config import Config
from codefinetuner.convert.run import (
    _ensure_output_paths_exist,
    _sync_converter_script_version,
    _log_subprocess_output, 
    _convert_to_gguf, 
    run
)


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config 


# --- _ensure_output_paths_exist ---
 
def test_ensure_output_paths_exist_creates_gguf_parent_dir(config):
    _ensure_output_paths_exist(config)
    assert config.lora_model_gguf_path.parent.exists()


# --- _sync_converter_script_version ---

def test_sync_converter_script_version_skips_download_when_up_to_date(config, mocker):
    """No HTTP request is made when the script and version marker are both current."""
    script_path = config.convert_hf_to_gguf_local_path
    script_path.parent.mkdir(parents=True, exist_ok=True)  
    script_path.write_text("# script", encoding="utf-8")
    
    version_marker = script_path.with_suffix(".version")
    version_marker.write_text("0.9.1")
    
    mocker.patch("importlib.metadata.version", return_value="0.9.1")
    http_mock = mocker.patch("codefinetuner.convert.run.httpx.Client")

    _sync_converter_script_version(config)

    http_mock.assert_not_called()
 
 
def test_sync_converter_script_version_downloads_when_old_version(config, mocker):
    """HTTP GET is issued when the local script file is outdated"""
    script_path = config.convert_hf_to_gguf_local_path
    script_path.parent.mkdir(parents=True, exist_ok=True)  
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
    mocker.patch("codefinetuner.convert.run.httpx.Client", return_value=mock_client)

    _sync_converter_script_version(config)

    mock_client.get.assert_called_once()
    assert config.convert_hf_to_gguf_local_path.read_text(encoding="utf-8") == "# downloaded script"


# --- _log_subprocess_output ---

def test_log_subprocess_output_logs_info_line(mocker):
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter(["INFO: model loaded successfully\n"])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    mock_logger.info.assert_called_once()


def test_log_subprocess_output_logs_warning_line(mocker):
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter(["WARNING: low memory\n"])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    mock_logger.warning.assert_called_once()


def test_log_subprocess_output_logs_error_line(mocker):
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter(["ERROR: conversion failed\n"])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    mock_logger.error.assert_called_once()


def test_log_subprocess_output_empty_stdout_logs_nothing(mocker):
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter([])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    mock_logger.info.assert_not_called()
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()


def test_log_subprocess_output_flushes_multiline_block_as_single_message(mocker):
    """Consecutive non-header lines belonging to one log entry are flushed together."""
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter([
        "INFO: starting conversion\n",
        "  continuing info line\n",
        "INFO: done\n",
    ])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    # Two INFO: entries → two info calls
    assert mock_logger.info.call_count == 2


# --- _convert_to_gguf ---

def test_convert_to_gguf_calls_popen(config, mocker):
    mock_popen = mocker.patch("codefinetuner.convert.run.subprocess.Popen")
    mock_process = mocker.Mock()
    mock_process.stdout = iter([])
    mock_process.returncode = 0
    mock_popen.return_value = mock_process
    mocker.patch("codefinetuner.convert.run._log_subprocess_output")

    _convert_to_gguf(config)

    mock_popen.assert_called_once()


def test_convert_to_gguf_raises_on_nonzero_returncode(config, mocker):
    mock_popen = mocker.patch("codefinetuner.convert.run.subprocess.Popen")
    mock_process = mocker.Mock()
    mock_process.stdout = iter([])
    mock_process.returncode = 1
    mock_popen.return_value = mock_process
    mocker.patch("codefinetuner.convert.run._log_subprocess_output")

    with pytest.raises(RuntimeError, match="exit code"):
        _convert_to_gguf(config)


# --- run ---

def test_run(config, mocker):
    mocker.patch("codefinetuner.convert.run._ensure_output_paths_exist")
    mocker.patch("codefinetuner.convert.run._sync_converter_script_version")
    convert_mock = mocker.patch("codefinetuner.convert.run._convert_to_gguf")

    run(config)

    convert_mock.assert_called_once_with(config)
