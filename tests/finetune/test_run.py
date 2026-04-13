import pathlib

import pytest
from datasets import IterableDataset

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config
from codefinetuner.finetune.run import (
    _ensure_clean_checkpoint_dir,
    load_datasets,
    run
)


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML and redirect paths to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- _ensure_clean_checkpoint_dir ---

def test_ensure_clean_checkpoint_dir_preserves_checkpoint_when_no_clear(config, tmp_path):
    config.trainer_resume_from_checkpoint = "chechpoint-test"
    config.trainer_clear_checkpoint_dir = False
    config.trainer_checkpoints_dir_path = tmp_path
    test_checkpoint = config.trainer_checkpoints_dir_path / "checkpoint-test"
    test_checkpoint.mkdir(parents=True, exist_ok=True)
 
    _ensure_clean_checkpoint_dir(config)
 
    assert test_checkpoint.exists()


def test_ensure_clean_checkpoint_dir_clear_removes_checkpoint_and_recreates_dir(config, tmp_path):
    config.trainer_resume_from_checkpoint = None
    config.trainer_clear_checkpoint_dir = True
    config.trainer_checkpoints_dir_path = tmp_path
    config.trainer_checkpoints_dir_path = tmp_path
    test_checkpoint = config.trainer_checkpoints_dir_path / "checkpoint-test"
    test_checkpoint.mkdir(parents=True, exist_ok=True)
 
    _ensure_clean_checkpoint_dir(config)
 
    assert config.trainer_checkpoints_dir_path.exists()
    assert not test_checkpoint.exists()
    

def test_ensure_clean_checkpoint_dir_clear_recreates_dir(config, tmp_path):
    config.trainer_resume_from_checkpoint = None
    config.trainer_clear_checkpoint_dir = True
    config.trainer_checkpoints_dir_path = tmp_path
 
    _ensure_clean_checkpoint_dir(config)
 
    assert config.trainer_checkpoints_dir_path.exists()

 
def test_ensure_clean_checkpoint_dir_no_checkpoint_no_clear_preserves_existing(config, tmp_path):
    config.trainer_resume_from_checkpoint = None
    config.trainer_clear_checkpoint_dir = False
    config.trainer_checkpoints_dir_path = tmp_path
    test_checkpoint = config.trainer_checkpoints_dir_path / "checkpoint-test"
    test_checkpoint.mkdir(parents=True, exist_ok=True)
 
    _ensure_clean_checkpoint_dir(config)
 
    assert test_checkpoint.exists()
 
 
def test_ensure_clean_checkpoint_dir_raises_when_both_set(config, tmp_path):
    config.trainer_resume_from_checkpoint = "last"
    config.trainer_clear_checkpoint_dir = True
    config.trainer_checkpoints_dir_path = tmp_path
 
    with pytest.raises(RuntimeError):
        _ensure_clean_checkpoint_dir(config)


# --- load_datasets ---
 
def test_load_datasets_returns_two_iterable_datasets(config):
    train_ds, eval_ds = load_datasets(config)
    assert isinstance(train_ds, IterableDataset)
    assert isinstance(eval_ds, IterableDataset)
 
 
def test_load_datasets_train_has_required_keys(config):
    train_dataset, _ = load_datasets(config)
    first_example = next(iter(train_dataset))
    assert "input_ids" in first_example 
    assert "attention_mask" in first_example
    assert "labels" in first_example
 
 
def test_load_datasets_eval_has_required_keys(config):
    _, eval_dataset = load_datasets(config)
    first_example = next(iter(eval_dataset))
    assert "input_ids" in first_example
    assert "attention_mask" in first_example
    assert "labels" in first_example


# --- run ---

def test_run(config, mocker):
    # Patch all internal function calls
    ensure_clean_checkpoint_dir_mock = mocker.patch("codefinetuner.finetune.run._ensure_clean_checkpoint_dir")
    load_datasets_mock = mocker.patch("codefinetuner.finetune.run.load_datasets", return_value=(mocker.Mock(), mocker.Mock()))
    load_and_configure_lora_model_mock = mocker.patch("codefinetuner.finetune.run.load_and_configure_lora_model")
    tokenizer_mock = mocker.patch("codefinetuner.finetune.run.AutoTokenizer.from_pretrained")
    train_mock = mocker.patch("codefinetuner.finetune.run.train_lora_model", return_value=[{"loss": 0.5}])
    merge_lora_and_save_mock = mocker.patch("codefinetuner.finetune.run.merge_lora_and_save")
    save_log_mock = mocker.patch("codefinetuner.finetune.run.save_log")
    plot_loss_mock = mocker.patch("codefinetuner.finetune.run.plot_loss")

    run(config)

    ensure_clean_checkpoint_dir_mock.assert_called_once()
    load_datasets_mock.assert_called_once()
    load_and_configure_lora_model_mock.assert_called_once()
    tokenizer_mock.assert_called_once()
    merge_lora_and_save_mock.assert_called_once()
    save_log_mock.assert_called_once_with(config, [{"loss": 0.5}])
    plot_loss_mock.assert_called_once()
 