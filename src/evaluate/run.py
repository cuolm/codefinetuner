import argparse
import logging
import logging.config

from transformers.trainer_utils import get_last_checkpoint

from .analyze import calculate_metric, save_all_metric_stats, plot_metric_and_save, plot_metric_averages_and_save
from .config import Config
from .evaluate import evaluate_and_save
from .generate import generate_and_save
from .benchmark import create_benchmark_dataset


logger = logging.getLogger("src.evaluate.run")


def _setup_logger(log_level: str) -> None:
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
            "src.evaluate": {  
                "handlers": ["stderr_handler"],
                "level": log_level,
                "propagate": False,
            }
        }
    }
    logging.config.dictConfig(logger_config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        metavar="NAME_OR_LAST",
        help='Checkpoint name, or "last"'
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,  
        help="Skip generation, use existing data to update plots"
    )
    parser.add_argument(
        "--overwrite-dataset",
        action="store_true",
        default=False,  
        help="Overwrite benchmark dataset"
    )
    return parser.parse_args()


def main():
    config = Config()
    user_args = _parse_args()
    _setup_logger(user_args.log_level)

    if user_args.overwrite_dataset or not config.benchmark_dataset_path.exists():
        dataset_len = create_benchmark_dataset(config)
        if checkpoint_path is None:
            logger.error(f"No checkpoint found under {config.trainer_output_dir_path}")
            raise RuntimeError("No checkpoint available for evaluation.")
        logger.info(f"Created new benchmark dataset '{config.benchmark_dataset_path}' with '{dataset_len}' examples")
    else:
        logger.info(f"Proceeding with existing file '{config.benchmark_dataset_path}'...")
    
    if not user_args.plot_only: 
        if user_args.checkpoint == "last":
            checkpoint_path = get_last_checkpoint(config.trainer_output_dir_path)
        else:
            checkpoint_path = config.trainer_output_dir_path / user_args.checkpoint
        generate_and_save(config, checkpoint_path)
        evaluate_and_save(config)

    metric_configurations = [
        (config.sentencebleu_metric_name, config.sentencebleu_plot_path, True),
        (config.codebleu_metric_name, config.codebleu_plot_path, True),
        (config.exact_match_metric_name, config.exact_match_plot_path, True),
        (config.line_match_metric_name, config.line_match_plot_path, True),
        (config.perplexity_name, config.perplexity_plot_path, False),
    ]

    all_metric_stats_np = []
    for metric_name, plot_path, higher_is_better in metric_configurations:
        metric_stats_np = calculate_metric(config, metric_name, higher_is_better)
        plot_metric_and_save(metric_stats_np, metric_name, plot_path)
        all_metric_stats_np.append(metric_stats_np)
    
    save_all_metric_stats(config, user_args.checkpoint, all_metric_stats_np)
    plot_metric_averages_and_save(config.benchmark_evaluation_report_path, config.benchmark_evaluation_averages_path) 


if __name__ == "__main__":
    main()