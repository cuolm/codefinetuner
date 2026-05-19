import argparse
import math
import logging.config
import shutil
import sys
from typing import Tuple
from pathlib import Path

from datasets import Features, IterableDataset, Sequence, Value, load_dataset
from transformers import AutoTokenizer

from .config import Config
from .model import load_and_configure_lora_model
from .train import train_lora_model, save_log, select_checkpoint_and_save, merge_lora_and_save, plot_loss


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
    parser = argparse.ArgumentParser(description="Start or resume LoRA model training")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML config file path with 'finetune:' section. All Config.MISSING fields must be provided."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()
    return args


def _ensure_output_paths_exist(config) -> None:
    paths = [
        config.finetune_outputs_dir_path,
        config.trainer_checkpoints_dir_path,
        config.trainer_model_merge_offload_folder_path,
        config.trainer_log_path, 
        config.trainer_plot_path,
        config.selected_checkpoint_path,
        config.lora_model_path
    ]
    for path in paths:
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created parent directory: {path.parent}")
        else:
            logger.debug(f"Parent directory already exists: {path.parent}") 


def _ensure_clean_checkpoint_dir(config: Config) -> None:
    checkpoints_dir = config.trainer_checkpoints_dir_path
    checkpoint = config.trainer_resume_from_checkpoint
    clear = config.trainer_clear_checkpoint_dir
    if checkpoint and not clear:
        logger.info(f"Resuming from: {checkpoint}")
    elif not checkpoint and clear: 
        if checkpoints_dir.exists():
            logger.warning(f"Deleting: {checkpoints_dir}")
            shutil.rmtree(checkpoints_dir)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cleared: {checkpoints_dir}")
    elif checkpoint and clear: 
        raise RuntimeError("Configuration conflict: Cannot resume from a checkpoint while '--delete-all-checkpoints' is set.")
    else:
        logger.info(f"Starting fresh training. Existing checkpoints in {checkpoints_dir} are preserved.")


def load_datasets(config: Config) -> Tuple[IterableDataset, IterableDataset]:
    # Define the expected schema/features of datasets.
    # Use 'int32' which the datasets library and pytorch map correctly to int tensors.
    dataset_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'attention_mask': Sequence(feature=Value(dtype='int32')),
        'labels': Sequence(feature=Value(dtype='int32')),
    })

    # Enable streaming mode to load the dataset as an iterator.
    # This allows processing data samples on-the-fly without downloading or loading the entire dataset into memory.
    # https://huggingface.co/docs/datasets/stream
    train_dataset = load_dataset("json", data_files=str(config.train_dataset_path), features=dataset_features, streaming=True)["train"]
    train_dataset = train_dataset.shuffle(buffer_size=config.dataset_shuffle_buffer_size, seed=config.dataset_shuffle_seed)  # take up to suffle_buffer_size examples and randomly shuffle them
    eval_dataset = load_dataset("json", data_files=str(config.eval_dataset_path), features=dataset_features, streaming=True)["train"]
    return train_dataset, eval_dataset 


def _get_dataset_length(path: Path) -> int:
    """
    Calculates line count for streaming dataset progress estimation.
    We need to do this because we load it as a streaming dataset iterator, which can only be iterated once.
    """
    if not path.exists():
        raise FileNotFoundError( f"Training dataset not found at expected path: {path}. Ensure the dataset file exists before initializing the Config.")
    count = 0
    with open(path, "r", encoding="utf-8") as file:
        for _ in file:
            count += 1
    logger.debug(f"Dataset length for '{path.name}': {count} lines")
    return count


def _calculate_max_steps(config, train_dataset_lenght) -> int:
    """
    Calculates total steps based on effective batch size and dataset length.
    Because we use streaming dataset iterators for efficiency, we cannot use num_train_epochs trainer class parameter directly.
    Instead, we need to calculate max_steps and pass it to the trainer.
    """
    if train_dataset_lenght == 0:
        logger.warning("Dataset length is 0; max_steps will be 0.")
        return 0
        
    effective_batch_size = (config.trainer_per_device_train_batch_size * config.trainer_gradient_accumulation_steps)
    if effective_batch_size == 0:
        raise ValueError("Effective batch size (batch_size * grad_accum) cannot be zero.")
    steps_per_epoch = math.ceil(config.dataset_train_dataset_length / effective_batch_size)
    max_steps = steps_per_epoch * config.trainer_num_train_epochs
    logger.debug(f"Calculated training schedule: {max_steps} total steps ({steps_per_epoch} steps/epoch for {config.trainer_num_train_epochs} epochs)")
    return max_steps
    

def run(config: Config) -> None:
    _ensure_output_paths_exist(config)
    _ensure_clean_checkpoint_dir(config)

    train_dataset_length = _get_dataset_length(config.train_dataset_path)
    trainer_max_steps = _calculate_max_steps(config, train_dataset_length)

    train_dataset, eval_dataset = load_datasets(config)
    logger.info(f"Dataset: {train_dataset_length} train examples, max_steps={trainer_max_steps}")

    lora_model = load_and_configure_lora_model(config)
    lora_model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"

    log_history = train_lora_model(config, lora_model, tokenizer, train_dataset, eval_dataset, trainer_max_steps)
    save_log(config, log_history)
    select_checkpoint_and_save(config)
    merge_lora_and_save(config, tokenizer)
    plot_loss(config)


def main() -> None:
    user_args = _parse_args()
    _setup_logger(user_args.log_level)
    try:
        finetune_config = Config.load_from_yaml(user_args.config)
        run(finetune_config)
    except Exception as e:
        logger.exception(f"Finetuning failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
