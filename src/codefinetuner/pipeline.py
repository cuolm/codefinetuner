import argparse
import logging
import logging.config
import sys
from pathlib import Path

from .preprocess.config import Config as PreprocessConfig
from .preprocess.run import run as preprocess_run
from .finetune.config import Config as FinetuneConfig
from .finetune.run import run as finetune_run
from .evaluate.config import Config as EvaluateConfig
from .evaluate.run import run as evaluate_run
from .convert.config import Config as ConvertConfig
from .convert.run import run as convert_run


logger = logging.getLogger(__name__)


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
        },
    }
    logging.config.dictConfig(logger_config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full pipeline: preprocess -> finetune -> evaluate."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file path with 'preprocess:', 'finetune:' and 'evaluate:' sections.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for the pipeline entrypoint.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocess stage and reuse existing datasets.",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip finetune stage and reuse existing checkpoints.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip evaluation stage.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion to gguf file stage.",
    )
    return parser.parse_args()


def run_pipeline(
    config_path: Path | str,
    skip_preprocess: bool = False,
    skip_finetune: bool = False,
    skip_evaluate: bool = False,
    skip_convert: bool = False,
) -> None:
    logger.info("Starting end-to-end pipeline")

    config_path = Path(config_path)

    if not skip_preprocess:
        logger.info("=== Stage 1/4: Preprocess ===")
        preprocess_config = PreprocessConfig.load_from_yaml(config_path)
        preprocess_run(preprocess_config)
        logger.info("Finished preprocess stage")
    else:
        logger.info("Skipping preprocess stage")

    if not skip_finetune:
        logger.info("=== Stage 2/4: Finetune ===")
        finetune_config = FinetuneConfig.load_from_yaml(config_path) 
        finetune_run(finetune_config)
        logger.info("Finished finetune stage")
    else:
        logger.info("Skipping finetune stage")

    if not skip_evaluate:
        logger.info("=== Stage 3/4: Evaluate ===")
        evaluate_config = EvaluateConfig.load_from_yaml(config_path)
        evaluate_run(evaluate_config)
        logger.info("Finished evaluate stage")
    else:
        logger.info("Skipping evaluate stage")
    
    if not skip_convert:
        logger.info("=== Stage 4/4: Convert ===")
        convert_config = ConvertConfig.load_from_yaml(config_path)
        convert_run(convert_config)
        logger.info("Finished conversion stage")
    else:
        logger.info("Skipping conversion stage")

    logger.info("Pipeline completed successfully")


def main() -> None:
    user_args = _parse_args()
    _setup_logger(user_args.log_level)
    try:
        run_pipeline(
            config_path=user_args.config,
            skip_preprocess=user_args.skip_preprocess,
            skip_finetune=user_args.skip_finetune,
            skip_evaluate=user_args.skip_evaluate,
            skip_convert=user_args.skip_convert,
        )
    except Exception:
        logger.exception("Pipeline execution failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
