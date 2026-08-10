import contextlib
import pathlib
import textwrap
from dataclasses import dataclass, field, make_dataclass

import pytest

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner import tracking as mlf
from codefinetuner.tracking import TrackerConfig


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> TrackerConfig:
    """Load a TrackerConfig from the test YAML."""
    test_config = TrackerConfig.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config.mlflow_tracking_path = None  # set to none, recalculate it in _resolve_paths()
    test_config._resolve_paths()  # regenerates paths relative to the new workspace_path
    return test_config


@dataclass
class _DummyStageConfig:
    learning_rate: float = 5e-5
    epochs: int = 3
    model_name: str = "unsloth/qwen2.5-coder-3b"
    rng: object = None  # should be skipped via _SKIP_FIELDS


# --- load_from_yaml ---

def test_load_from_yaml_success(config):
    assert config.mlflow_experiment_name == "codefinetuner"
    assert config.use_mlflow is True


def test_load_from_yaml_missing_file(tmp_path):
    nonexistent_yaml = tmp_path / "nonexistent_yaml.yaml"
    with pytest.raises(FileNotFoundError):
        TrackerConfig.load_from_yaml(nonexistent_yaml)


def test_load_from_yaml_invalid_yaml(tmp_path):
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("tracking:\n  key: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse YAML file"):
        TrackerConfig.load_from_yaml(invalid_yaml)


def test_load_from_yaml_ignores_unknown_keys(tmp_path):
    """Extra YAML keys (e.g. from global anchors) must not raise."""
    config_text = textwrap.dedent("""
        tracking:
          workspace_path: "tests"
          use_mlflow: true
          mlflow_experiment_name: "my_experiment"
          unknown_key_that_does_not_exist: 999
    """)
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    test_config = TrackerConfig.load_from_yaml(test_config_path)
    assert test_config.mlflow_experiment_name == "my_experiment"


# --- _validate_dependencies ---

def test_validate_dependencies_raises_when_mlflow_unavailable(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", False)
    with pytest.raises(ImportError, match="MLflow tracking is enabled"):
        TrackerConfig(use_mlflow=True)


def test_validate_dependencies_ok_when_disabled(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", False)
    test_config = TrackerConfig(use_mlflow=False)
    assert test_config.use_mlflow is False


def test_validate_dependencies_ok_when_available(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    test_config = TrackerConfig(use_mlflow=True)
    assert test_config.use_mlflow is True


# --- _resolve_paths ---

def test_resolve_paths_defaults_workspace_to_cwd(tmp_path, mocker):
    mocker.patch("pathlib.Path.cwd", return_value=tmp_path)
    test_config = TrackerConfig()
    assert test_config.workspace_path == tmp_path


def test_resolve_paths_defaults_mlflow_tracking_path(config, tmp_path):
    assert config.mlflow_tracking_path == tmp_path / "outputs" / "mlflow"


def test_resolve_paths_creates_tracking_dir(config):
    assert config.mlflow_tracking_path.exists()


def test_resolve_paths_respects_custom_tracking_path(tmp_path):
    custom_path = tmp_path / "custom_mlflow"
    test_config = TrackerConfig(mlflow_tracking_path=custom_path)
    assert test_config.mlflow_tracking_path == custom_path
    assert custom_path.exists()


# --- start_run ---

def test_start_run_returns_nullcontext_when_disabled(config, mocker):
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    config.use_mlflow = False

    ctx = mlf.start_run(config, run_name="test-run")

    mlflow_mock.start_run.assert_not_called()
    assert isinstance(ctx, contextlib.nullcontext)


def test_start_run_returns_nullcontext_when_active_run_exists(config, mocker):
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()
    config.use_mlflow = True

    ctx = mlf.start_run(config, run_name="test-run")

    mlflow_mock.start_run.assert_not_called()
    assert isinstance(ctx, contextlib.nullcontext)


def test_start_run_initializes_mlflow_correctly(config, mocker):
    """Verify the complete initialization sequence and path formatting for a new run."""
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = None
    mlflow_mock.get_experiment_by_name.return_value = None
    config.use_mlflow = True

    mlf.start_run(config, run_name="test-run")

    # Assert correct path/URI transformations
    expected_db_path = (config.mlflow_tracking_path / "mlflow.db").resolve().as_posix()
    expected_artifact_dir = (config.mlflow_tracking_path / "artifacts").resolve().as_uri()

    # Verify the exact sequence of required configuration calls
    mlflow_mock.set_tracking_uri.assert_called_once_with(f"sqlite:///{expected_db_path}")
    mlflow_mock.create_experiment.assert_called_once_with(
        name=config.mlflow_experiment_name, artifact_location=expected_artifact_dir
    )
    mlflow_mock.set_experiment.assert_called_once_with(config.mlflow_experiment_name)
    mlflow_mock.start_run.assert_called_once_with(run_name="test-run")


# --- _flatten_fields ---

def test_flatten_prefixes_keys():
    flat = mlf._flatten_fields(_DummyStageConfig(), prefix="finetune")
    assert set(flat.keys()) == {"finetune.learning_rate", "finetune.epochs", "finetune.model_name"}


def test_flatten_skips_configured_fields():
    flat = mlf._flatten_fields(_DummyStageConfig(), prefix="finetune")
    assert "finetune.rng" not in flat


def test_flatten_stringifies_non_primitive_values():
    @dataclass
    class NestedConfig:
        path: pathlib.Path = pathlib.Path("/tmp/test")

    flat = mlf._flatten_fields(NestedConfig(), prefix="stage")
    assert flat["stage.path"] == str(pathlib.Path("/tmp/test"))


def test_flatten_preserves_primitive_types():
    flat = mlf._flatten_fields(_DummyStageConfig(), prefix="finetune")
    assert isinstance(flat["finetune.learning_rate"], float)
    assert isinstance(flat["finetune.epochs"], int)
    assert isinstance(flat["finetune.model_name"], str)


# --- log_stage_params ---

def test_log_stage_params_skipped_when_mlflow_unavailable(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", False)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlf.log_stage_params(_DummyStageConfig(), "finetune")
    mlflow_mock.log_params.assert_not_called()


def test_log_stage_params_skipped_when_no_active_run(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = None
    mlf.log_stage_params(_DummyStageConfig(), "finetune")
    mlflow_mock.log_params.assert_not_called()


def test_log_stage_params_logs_flattened_params(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    mlf.log_stage_params(_DummyStageConfig(), "finetune")

    mlflow_mock.log_params.assert_called_once_with({
        "finetune.learning_rate": 5e-5,
        "finetune.epochs": 3,
        "finetune.model_name": "unsloth/qwen2.5-coder-3b",
    })


def test_log_stage_params_batches_over_100_pairs(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    BigStageConfig = make_dataclass(
        "BigStageConfig",
        [(f"param_{i}", float, field(default=float(i))) for i in range(150)],
    )

    mlf.log_stage_params(BigStageConfig(), "stage")

    assert mlflow_mock.log_params.call_count == 2
    first_call_size = len(mlflow_mock.log_params.call_args_list[0].args[0])
    second_call_size = len(mlflow_mock.log_params.call_args_list[1].args[0])
    assert first_call_size == 100
    assert second_call_size == 50


# --- log_metrics ---

def test_log_metrics_skipped_when_mlflow_unavailable(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", False)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlf.log_metrics({"loss": 0.5})
    mlflow_mock.log_metrics.assert_not_called()


def test_log_metrics_skipped_when_no_active_run(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = None
    mlf.log_metrics({"loss": 0.5})
    mlflow_mock.log_metrics.assert_not_called()


def test_log_metrics_logs_with_step(mocker):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    mlf.log_metrics({"loss": 0.5}, step=10)

    mlflow_mock.log_metrics.assert_called_once_with({"loss": 0.5}, step=10)


# --- log_artifact ---

def test_log_artifact_skipped_when_mlflow_unavailable(mocker, tmp_path):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", False)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    dummy_file = tmp_path / "file.txt"
    dummy_file.write_text("data")
    mlf.log_artifact(dummy_file)
    mlflow_mock.log_artifact.assert_not_called()


def test_log_artifact_skipped_when_no_active_run(mocker, tmp_path):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = None
    dummy_file = tmp_path / "file.txt"
    dummy_file.write_text("data")
    mlf.log_artifact(dummy_file)
    mlflow_mock.log_artifact.assert_not_called()


def test_log_artifact_logs_existing_file(mocker, tmp_path):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    dummy_file = tmp_path / "file.txt"
    dummy_file.write_text("data")

    mlf.log_artifact(dummy_file, artifact_path="configs")

    mlflow_mock.log_artifact.assert_called_once_with(str(dummy_file), artifact_path="configs")


def test_log_artifact_warns_when_file_missing(mocker, tmp_path, caplog):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    missing_file = tmp_path / "missing.txt"

    with caplog.at_level("WARNING"):
        mlf.log_artifact(missing_file)

    mlflow_mock.log_artifact.assert_not_called()
    assert "does not exist" in caplog.text


# --- log_model_artifacts ---

def test_log_model_artifacts_skipped_when_mlflow_unavailable(mocker, tmp_path):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", False)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    dummy_file = tmp_path / "dir"
    dummy_file.mkdir()
    mlf.log_model_artifacts(dummy_file)
    mlflow_mock.log_artifacts.assert_not_called()


def test_log_model_artifacts_skipped_when_no_active_run(mocker, tmp_path):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = None
    dummy_file = tmp_path / "dir"
    dummy_file.mkdir()
    mlf.log_model_artifacts(dummy_file)
    mlflow_mock.log_artifacts.assert_not_called()


def test_log_model_artifacts_logs_existing_file(mocker, tmp_path):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    dummy_file = tmp_path / "dir"
    dummy_file.mkdir()

    mlf.log_model_artifacts(dummy_file, artifact_path="configs")

    mlflow_mock.log_artifacts.assert_called_once_with(str(dummy_file), artifact_path="configs")


def test_log_model_artifacts_warns_when_file_missing(mocker, tmp_path, caplog):
    mocker.patch("codefinetuner.tracking._MLFLOW_AVAILABLE", True)
    mlflow_mock = mocker.patch("codefinetuner.tracking.mlflow")
    mlflow_mock.active_run.return_value = mocker.Mock()

    missing_file = tmp_path / "missing_dir"

    with caplog.at_level("WARNING"):
        mlf.log_model_artifacts(missing_file)

    mlflow_mock.log_artifacts.assert_not_called()
    assert "does not exist" in caplog.text
