import contextlib
import logging
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

_SKIP_FIELDS = {
    "tree_sitter_parser", "tree_sitter_block_types", "tree_sitter_subblock_types",
    "rng", "device", "model_dtype",
}


@dataclass
class TrackerConfig:
    use_mlflow: bool = False
    mlflow_experiment_name: str = "codefinetuner"

    workspace_path: Path | None = None
    mlflow_tracking_path: Path | None = None 
    mlflow_model_logging_strategy: str = "none"

    @classmethod
    def load_from_yaml(cls, yaml_path: Path) -> "TrackerConfig":
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")

        config_dict = OmegaConf.structured(cls)
        try:
            yaml_file_node = OmegaConf.load(yaml_path)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file: {yaml_path}") from e

        yaml_file_dict = OmegaConf.to_container(yaml_file_node, resolve=True)
        yaml_tracking_dict = yaml_file_dict.get("tracking", {})

        yaml_tracking_valid_dict = {}
        for f in fields(cls):
            if f.name in yaml_tracking_dict:
                yaml_tracking_valid_dict[f.name] = yaml_tracking_dict[f.name]
        logger.debug(f"Filtered YAML tracking configuration: {yaml_tracking_valid_dict}")

        merged_config_dict = OmegaConf.merge(config_dict, yaml_tracking_valid_dict)
        return OmegaConf.to_object(merged_config_dict)

    def __post_init__(self) -> None:
        self._validate_dependencies()
        self._resolve_paths()

    def _validate_dependencies(self) -> None:
        if self.use_mlflow and not _MLFLOW_AVAILABLE:
            raise ImportError(
                "\n[ERROR] MLflow tracking is enabled in your configuration, but the package is not installed.\n"
                "To resolve this, install the package with mlflow support:\n"
                "  - Using uv:       uv add \"codefinetuner[mlflow]\"\n"
                "  - Using pip:      pip install \"codefinetuner[mlflow]\"\n"
                "  - For dev setup:  uv add mlflow --optional mlflow"
            )

    def _resolve_paths(self) -> None:
        if self.workspace_path is None:
            self.workspace_path = Path.cwd()
        else:
            self.workspace_path = Path(self.workspace_path)

        if self.mlflow_tracking_path is None:
            self.mlflow_tracking_path = self.workspace_path / "outputs" / "mlflow"
        else:
            self.mlflow_tracking_path = Path(self.mlflow_tracking_path)

        self.mlflow_tracking_path.mkdir(parents=True, exist_ok=True)


def start_run(config: TrackerConfig, run_name: str):
    if not config.use_mlflow:
        return contextlib.nullcontext()
    if mlflow.active_run() is not None:
        return contextlib.nullcontext()
        
    # Setup full relational database tracking local backend
    db_file_path = (config.mlflow_tracking_path / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_file_path.as_posix()}")
    
    # Configure explicit local folder location for heavy binary artifacts
    experiment_name = config.mlflow_experiment_name
    if mlflow.get_experiment_by_name(experiment_name) is None:
        artifact_dir = (config.mlflow_tracking_path / "artifacts").resolve()
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_dir.as_uri())
        
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def _flatten(config, prefix: str) -> dict[str, Any]:
    flat = {}
    for f in fields(config):
        if f.name in _SKIP_FIELDS:
            continue
        value = getattr(config, f.name)
        flat[f"{prefix}.{f.name}"] = value if isinstance(value, (int, float, bool, str)) else str(value)
    return flat


def log_stage_params(stage_config, stage: str) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
    flat = _flatten(stage_config, prefix=stage)
    items = list(flat.items())
    for i in range(0, len(items), 100):  # mlflow caps log_params at 100 pairs/call
        mlflow.log_params(dict(items[i : i + 100]))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path, artifact_path: str | None = None) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
    path = Path(path)
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    else:
        logger.warning(f"log_artifact skipped, path does not exist: {path}")


def log_model_artifacts(path: Path, artifact_path: str | None = None) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
    path = Path(path)
    if path.exists():
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)
    else:
        logger.warning(f"log_model_artifacts skipped, path does not exist: {path}")
