import contextlib
import json
import logging
from dataclasses import dataclass, fields
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


def _flatten_fields(config, prefix: str) -> dict[str, Any]:
    flattened_dict = {}
    for field in fields(config):
        if field.name in _SKIP_FIELDS:
            continue
        field_value = getattr(config, field.name)
        key = f"{prefix}.{field.name}"
        if isinstance(field_value, (int, float, bool, str)):
            flattened_dict[key] = field_value
        else:
            flattened_dict[key] = str(field_value)
    return flattened_dict


def log_stage_params(stage_config, stage: str) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
    flattened_fields = _flatten_fields(stage_config, prefix=stage)
    items = list(flattened_fields.items())
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


def log_preprocess(preprocess_config: Any) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
    
    log_stage_params(preprocess_config, "preprocess")

    outputs_dir = preprocess_config.outputs_dir_path
    if outputs_dir.exists():
         mlflow.log_artifacts(str(outputs_dir), artifact_path="preprocess_outputs")


def log_finetune(finetune_config: Any, tracker_config: TrackerConfig) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
        
    log_stage_params(finetune_config, "finetune")

    results_dir = finetune_config.results_dir_path
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_file():
                mlflow.log_artifact(str(item), artifact_path="finetune_outputs")

    strategy = tracker_config.mlflow_model_logging_strategy
    if strategy in ("adapter", "all"):
        checkpoint_path = finetune_config.selected_checkpoint_path
        if checkpoint_path.exists():
            logger.info("Logging LoRA adapter artifact to MLflow")
            mlflow.log_artifacts(str(checkpoint_path), artifact_path="finetune_outputs/lora_adapter")


def log_evaluate(evaluate_config: Any) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
        
    log_stage_params(evaluate_config, "evaluate")

    analysis_results_path = evaluate_config.analysis_results_path
    if analysis_results_path.exists():
        with open(analysis_results_path, "r") as results_file:
            report = json.load(results_file)
            
        for stat in report.get("all_metric_stats", []):
            metric_name = stat["metric"]
            base_avg = stat["base_average"]
            lora_avg = stat["lora_average"]
            
            if stat.get("higher_is_better", True):
                improvement = lora_avg - base_avg
            else:
                improvement = base_avg - lora_avg
                
            mlflow.log_metrics({
                f"evaluate.{metric_name}.base": float(base_avg),
                f"evaluate.{metric_name}.lora": float(lora_avg),
                f"evaluate.{metric_name}.improvement": float(improvement),
            })
            
    evaluation_outputs_dir = evaluate_config.outputs_dir_path
    if evaluation_outputs_dir.exists():
        mlflow.log_artifacts(str(evaluation_outputs_dir), artifact_path="evaluation_outputs")


def log_convert(convert_config: Any, tracker_config: TrackerConfig) -> None:
    if not _MLFLOW_AVAILABLE or mlflow.active_run() is None:
        return
        
    log_stage_params(convert_config, "convert")

    strategy = tracker_config.mlflow_model_logging_strategy
    if strategy in ("gguf", "all"):
        gguf_path = convert_config.lora_model_gguf_path
        if gguf_path.exists():
            logger.info("Logging final merged GGUF model artifact to MLflow")
            mlflow.log_artifacts(str(gguf_path), artifact_path="convert_outputs/finetuned_model_gguf")
