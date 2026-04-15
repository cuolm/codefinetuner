import argparse
import logging
import sys
import logging.config
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint

from .config import Config
from .benchmark import create_benchmark_dataset
from .generate import generate_and_save
from .evaluate import evaluate_and_save
from .analyze import analyze_metric, save_all_metric_stats, get_plot_path, plot_metric_and_save, plot_all_metric_averages_and_save


logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the pipeline YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def _setup_logger(log_level: str) -> None:
    """
    Configure the root logger ("") to capture logs from the entry point (__main__),
    all internal sub-modules, and third-party libraries via a single handler.
    """
    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "handlers": {
            "stderr_handler": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
            }
        },
        "loggers": {
            "": {  # "" corresponds to root logger 
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            }
        }
    }
    logging.config.dictConfig(logger_config)


def _silence_noisy_third_party_loggers() -> None:
    """Reduce log level of known noisy third‑party loggers."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    logging.getLogger("nltk").setLevel(logging.WARNING)
    # add more as needed


def _ensure_checkpoints(config: Config) -> None:
    checkpoints_dir = config.trainer_checkpoints_dir_path
    if not checkpoints_dir.exists():
        raise RuntimeError("Checkpoint directory not found.")
    if not list(checkpoints_dir.iterdir()):
        raise RuntimeError("No checkpoints in directory.")


def run(config: Config) -> None:
    if not config.benchmark_use_existing_dataset or not config.benchmark_dataset_path.exists():
        dataset_len = create_benchmark_dataset(config)
        logger.info(f"Created new benchmark dataset '{config.benchmark_dataset_path}' with '{dataset_len}' examples")
    else:
        logger.info(f"Proceeding with existing file '{config.benchmark_dataset_path}'...")
    
    if not config.plot_only: 
        _ensure_checkpoints(config)
        if config.trainer_checkpoint == "last":
            checkpoint_path = get_last_checkpoint(config.trainer_checkpoints_dir_path)
        else:
            checkpoint_path = config.trainer_checkpoints_dir_path / config.trainer_checkpoint 
        generate_and_save(config, checkpoint_path)
        evaluate_and_save(config)

    all_metric_stats_np = []
    for metric_name, higher_is_better in config.metric_configs:
        metric_stats_np = analyze_metric(config, metric_name, higher_is_better)
        plot_path = get_plot_path(config.benchmark_evaluation_results_dir, metric_name)
        plot_metric_and_save(metric_stats_np, metric_name, plot_path)
        all_metric_stats_np.append(metric_stats_np)
    
    all_metric_averages_plot_path = get_plot_path(config.benchmark_evaluation_results_dir, "all_metric_averages")
    plot_all_metric_averages_and_save(all_metric_stats_np, all_metric_averages_plot_path) 
    save_all_metric_stats(config, all_metric_stats_np)


def main():
    user_args = _parse_args()
    _setup_logger(user_args.log_level)
    _silence_noisy_third_party_loggers()
    try:
        evaluate_config = Config.load_from_yaml(user_args.config)
        run(evaluate_config)
    except Exception:
        logger.exception("Evaluation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
