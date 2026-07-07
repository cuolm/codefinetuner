import argparse
import logging.config
import sys
from pathlib import Path

from transformers import AutoTokenizer

from .config import Config
from .extract import get_code_blocks_from_auto_split, get_code_blocks_from_manual_split
from .process import create_fim_examples, estimate_bytes_per_token_ratio, tokenize_filter_and_save, augment_with_random_fim_examples
from .analyze import analyze_and_plot_datasets


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
        }
    }
    logging.config.dictConfig(logger_config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess code dataset for FIM fine-tuning.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file path with 'preprocess:' section. All Config.MISSING fields must be provided."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser.parse_args()


def _ensure_output_paths_exist(config) -> None:
    paths = [
        config.preprocess_outputs_dir_path,
        config.preprocess_results_path,
        config.train_dataset_path,
        config.eval_dataset_path,
        config.test_dataset_path,
        config.split_log_path
    ]

    for path in paths:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created parent directory: {path.parent}")
        else:
            logger.debug(f"Parent directory already exists: {path.parent}")  


def _clear_existing_datasets(config: Config) -> None:
    for file in [config.train_dataset_path, config.eval_dataset_path, config.test_dataset_path]:
        if file.exists():
            file.unlink()
            logger.info(f"Deleted old dataset: {file}")


def _validate_and_configure_tokenizer(config: Config, tokenizer: AutoTokenizer):
    required_tokens = [
        config.fim_prefix_token,
        config.fim_middle_token,
        config.fim_suffix_token,
        config.fim_pad_token,
        config.eos_token
    ]

    for token in required_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        
        if token_id == tokenizer.unk_token_id or token_id is None:
            raise ValueError(f"Token '{token}' defined in Config is missing from Tokenizer vocabulary.")

    tokenizer.pad_token = config.fim_pad_token
    tokenizer.eos_token = config.eos_token

    logger.info("Tokenizer validated and configured successfully.")


def run(config: Config) -> None:
    _ensure_output_paths_exist(config)
    _clear_existing_datasets(config)

    if config.split_mode == "auto":
        logger.info("Using auto-generated dataset split.")
        split_result= get_code_blocks_from_auto_split(config) 
    else: # config.split_mode == "manual":
        logger.info("Using manual dataset split from directories.")
        split_result = get_code_blocks_from_manual_split(config) 

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    _validate_and_configure_tokenizer(config, tokenizer)

    bytes_per_token_ratio = estimate_bytes_per_token_ratio(config, tokenizer, number_of_code_blocks=20000)
    logger.info(f"Estimated bytes_per_token_ratio: {bytes_per_token_ratio}")

    train_fim_examples_iter= create_fim_examples(config, split_result.train_iter, bytes_per_token_ratio)
    eval_fim_examples_iter = create_fim_examples(config, split_result.eval_iter, bytes_per_token_ratio)
    test_fim_examples_iter = create_fim_examples(config, split_result.test_iter, bytes_per_token_ratio)
    
    num_ast_train_examples = tokenize_filter_and_save(config, config.train_dataset_path, train_fim_examples_iter, tokenizer)
    num_ast_eval_examples = tokenize_filter_and_save(config, config.eval_dataset_path, eval_fim_examples_iter, tokenizer)
    num_ast_test_examples = tokenize_filter_and_save(config, config.test_dataset_path, test_fim_examples_iter, tokenizer)

    num_rand_train_examples = int(num_ast_train_examples * config.rand_to_ast_fim_examples_ratio)
    augment_with_random_fim_examples(config, tokenizer, split_result.train_paths, bytes_per_token_ratio, num_rand_train_examples, config.train_dataset_path)
    num_rand_eval_examples = int(num_ast_eval_examples * config.rand_to_ast_fim_examples_ratio)
    augment_with_random_fim_examples(config, tokenizer, split_result.eval_paths, bytes_per_token_ratio, num_rand_eval_examples, config.eval_dataset_path)
    num_rand_test_examples = int(num_ast_test_examples * config.rand_to_ast_fim_examples_ratio)
    augment_with_random_fim_examples(config, tokenizer, split_result.test_paths, bytes_per_token_ratio, num_rand_test_examples, config.test_dataset_path)

    fim_middle_token_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    eos_token_id = tokenizer.convert_tokens_to_ids(config.eos_token)

    analyze_and_plot_datasets(config, fim_middle_token_id, eos_token_id)

    logger.info("Saved train, eval, test datasets to disk")  


def main() -> None:
    user_args = _parse_args()
    _setup_logger(user_args.log_level)
    try:
        preprocess_config = Config.load_from_yaml(user_args.config)
        run(preprocess_config)
    except Exception:
        logger.exception(f"Preprocessing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
    