import pathlib

import pytest

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.convert.config import Config
from codefinetuner.convert.run import _log_subprocess_output, _convert_to_gguf, run


# --- Fixtures ---


@pytest.fixture
def config(mocker) -> Config:
    """
    Load a convert Config from the test YAML.
    _sync_converter_script_version is patched to prevent any network calls
    during construction.
    """
    mocker.patch.object(Config, "_sync_converter_script_version")
    return Config.load_from_yaml(test_config_path)


# --- _log_subprocess_output ---


def test_log_subprocess_output_logs_info_line(config, mocker):
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter(["INFO: model loaded successfully\n"])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    mock_logger.info.assert_called_once()


def test_log_subprocess_output_logs_warning_line(config, mocker):
    mock_logger = mocker.patch("codefinetuner.convert.run.logger")
    mock_process = mocker.Mock()
    mock_process.stdout = iter(["WARNING: low memory\n"])

    _log_subprocess_output(mock_process, prefix="llama.cpp")

    mock_logger.warning.assert_called_once()


def test_log_subprocess_output_logs_error_line(config, mocker):
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

def test_run_calls_convert_to_gguf(config, mocker):
    convert_mock = mocker.patch("codefinetuner.convert.run._convert_to_gguf")

    run(config)

    convert_mock.assert_called_once_with(config)
