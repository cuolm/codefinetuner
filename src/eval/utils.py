import argparse
import logging
import logging.config
from .prepare_benchmark import create_benchmark_dataset
from .config import Config

logger = logging.getLogger(__name__)

def _setup_logger(log_level: str) -> None:
    config = {
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
    logging.config.dictConfig(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate code comparison")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        metavar="NAME_OR_LAST",
                        help='Checkpoint name, or "last"')
    parser.add_argument("--plot-only",
                        action="store_true",
                        default=False,  
                        help="Skip generation, use existing data to update plots")
    parser.add_argument("--overwrite-dataset",
                        action="store_true",
                        default=False,  
                        help="Overwrite benchmark dataset")
    return parser.parse_args()


def _ensure_benchmark_dataset(config: Config, user_args: argparse.Namespace) -> None:
    if user_args.overwrite_dataset or not config.benchmark_dataset_path.exists():
        dataset_len = create_benchmark_dataset(config)
        logger.info(f"Created new benchmark dataset '{config.benchmark_dataset_path}' with '{dataset_len}' examples")
    else:
        logger.info(f"Proceeding with existing file '{config.benchmark_dataset_path}'...")


def _ensure_directories_exist(config: Config) -> None:
    """Create all required output directories."""
    directories = [
        config.base_results_tmp_path.parent,
        config.evaluation_results_path.parent,
    ]
    for dir_path in directories:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")