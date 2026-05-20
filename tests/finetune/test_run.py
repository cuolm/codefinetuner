import pathlib
import json

import pytest
from datasets import IterableDataset

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config
from codefinetuner.finetune.run import (
    _ensure_output_paths_exist,
    _ensure_clean_checkpoint_dir,
    load_datasets,
    _get_dataset_length,
    _calculate_max_steps,
    run
)


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load a Config from the test YAML and redirect paths to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config


@pytest.fixture
def config_with_datasets(config) -> Config:
    dummy_row = {
        "input_ids": [101, 102, 103],
        "attention_mask": [1, 1, 1],
        "labels": [101, 102, 103]
    }
    
    config.train_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    config.eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    
    config.train_dataset_path.write_text(json.dumps(dummy_row) + "\n", encoding="utf-8")
    config.eval_dataset_path.write_text(json.dumps(dummy_row) + "\n", encoding="utf-8")
    
    return config


# --- ensure_output_paths_exist ---

def test_ensure_output_paths_exist_creates_parent_dirs(config):
    _ensure_output_paths_exist(config)
    
    paths_to_check = [
        "finetune_outputs_dir_path",
        "trainer_checkpoints_dir_path",
        "trainer_model_merge_offload_folder_path",
        "trainer_log_path",
        "trainer_plot_path",
        "selected_checkpoint_path",
        "lora_model_path"
    ]
    
    for attr in paths_to_check:
        path = getattr(config, attr)
        assert path.parent.exists(), f"Parent directory for {attr} ({path.parent}) was not created."


# --- _ensure_clean_checkpoint_dir ---

def test_ensure_clean_checkpoint_dir_preserves_checkpoint_when_no_clear(config):
    config.trainer_resume_from_checkpoint = "chechpoint-test"
    config.trainer_clear_checkpoint_dir = False
    
    test_checkpoint = config.trainer_checkpoints_dir_path / "checkpoint-test"
    test_checkpoint.mkdir(parents=True, exist_ok=True)
 
    _ensure_clean_checkpoint_dir(config)
 
    assert test_checkpoint.exists()


def test_ensure_clean_checkpoint_dir_clear_removes_checkpoint_and_recreates_dir(config):
    config.trainer_resume_from_checkpoint = None
    config.trainer_clear_checkpoint_dir = True

    test_checkpoint = config.trainer_checkpoints_dir_path / "checkpoint-test"
    test_checkpoint.mkdir(parents=True, exist_ok=True)
 
    _ensure_clean_checkpoint_dir(config)
 
    assert config.trainer_checkpoints_dir_path.exists()
    assert not test_checkpoint.exists()
    

def test_ensure_clean_checkpoint_dir_clear_recreates_dir(config):
    config.trainer_resume_from_checkpoint = None
    config.trainer_clear_checkpoint_dir = True
 
    _ensure_clean_checkpoint_dir(config)
 
    assert config.trainer_checkpoints_dir_path.exists()

 
def test_ensure_clean_checkpoint_dir_no_checkpoint_no_clear_preserves_existing(config):
    config.trainer_resume_from_checkpoint = None
    config.trainer_clear_checkpoint_dir = False

    test_checkpoint = config.trainer_checkpoints_dir_path / "checkpoint-test"
    test_checkpoint.mkdir(parents=True, exist_ok=True)
 
    _ensure_clean_checkpoint_dir(config)
 
    assert test_checkpoint.exists()
 
 
def test_ensure_clean_checkpoint_dir_raises_when_both_set(config):
    config.trainer_resume_from_checkpoint = "last"
    config.trainer_clear_checkpoint_dir = True
 
    with pytest.raises(RuntimeError):
        _ensure_clean_checkpoint_dir(config)


# --- load_datasets ---
 
def test_load_datasets_returns_two_iterable_datasets(config_with_datasets):
    train_ds, eval_ds = load_datasets(config_with_datasets)
    assert isinstance(train_ds, IterableDataset)
    assert isinstance(eval_ds, IterableDataset)
 
 
def test_load_datasets_train_has_required_keys(config_with_datasets):
    train_dataset, _ = load_datasets(config_with_datasets)
    first_example = next(iter(train_dataset))
    assert "input_ids" in first_example 
    assert "attention_mask" in first_example
    assert "labels" in first_example
 
 
def test_load_datasets_eval_has_required_keys(config_with_datasets):
    _, eval_dataset = load_datasets(config_with_datasets)
    first_example = next(iter(eval_dataset))
    assert "input_ids" in first_example
    assert "attention_mask" in first_example
    assert "labels" in first_example


# --- _get_dataset_length ---

def test_get_dataset_length_counts_lines(config_with_datasets):
    length = _get_dataset_length(config_with_datasets.train_dataset_path)
    assert length == 1


def test_get_dataset_length_raises_not_found_error(config_with_datasets):
    config_with_datasets.train_dataset_path = config_with_datasets.train_dataset_path / "missing_dataset.jsonl"
    with pytest.raises(FileNotFoundError):
        _get_dataset_length(config_with_datasets.train_dataset_path)


# --- _calculate_max_steps ---

def test_calculate_max_steps_logic(config):
    config.trainer_per_device_train_batch_size = 2
    config.trainer_gradient_accumulation_steps = 2
    config.trainer_num_train_epochs = 3
    # Simulating the property that your math calculation relies on internally:
    config.dataset_train_dataset_length = 100 
    
    # Math: batch = 2 * 2 = 4. Steps per epoch = ceil(100 / 4) = 25. Max steps = 25 * 3 = 75.
    max_steps = _calculate_max_steps(config, train_dataset_length=100)
    assert max_steps == 75


def test_calculate_max_steps_zero_length(config):
    assert _calculate_max_steps(config, train_dataset_length=0) == 0


def test_calculate_max_steps_zero_batch_size_raises(config):
    config.trainer_per_device_train_batch_size = 0
    config.trainer_gradient_accumulation_steps = 2
    
    with pytest.raises(ValueError, match="cannot be zero"):
        _calculate_max_steps(config, train_dataset_length=100)


# --- run ---

def test_run(config, mocker):
    # Patch all internal function calls
    ensure_output_paths_exist_mock  = mocker.patch("codefinetuner.finetune.run._ensure_output_paths_exist")
    ensure_clean_checkpoint_dir_mock = mocker.patch("codefinetuner.finetune.run._ensure_clean_checkpoint_dir")
    get_length_mock = mocker.patch("codefinetuner.finetune.run._get_dataset_length", return_value=100)
    calculate_max_steps_mock = mocker.patch("codefinetuner.finetune.run._calculate_max_steps", return_value=75)
    load_datasets_mock = mocker.patch("codefinetuner.finetune.run.load_datasets", return_value=(mocker.Mock(), mocker.Mock()))
    load_and_configure_lora_model_mock = mocker.patch("codefinetuner.finetune.run.load_and_configure_lora_model")
    tokenizer_mock = mocker.patch("codefinetuner.finetune.run.AutoTokenizer.from_pretrained")
    train_mock = mocker.patch("codefinetuner.finetune.run.train_lora_model", return_value=[{"loss": 0.5}])
    save_log_mock = mocker.patch("codefinetuner.finetune.run.save_log")
    select_checkpoint_and_save_mock = mocker.patch("codefinetuner.finetune.run.select_checkpoint_and_save")
    merge_lora_and_save_mock = mocker.patch("codefinetuner.finetune.run.merge_lora_and_save")
    plot_loss_mock = mocker.patch("codefinetuner.finetune.run.plot_loss")

    run(config)

    ensure_output_paths_exist_mock.assert_called_once()
    ensure_clean_checkpoint_dir_mock.assert_called_once()
    get_length_mock.assert_called_once()
    calculate_max_steps_mock.assert_called_once()
    load_datasets_mock.assert_called_once()
    load_and_configure_lora_model_mock.assert_called_once()
    tokenizer_mock.assert_called_once()
    train_mock.assert_called_once()
    save_log_mock.assert_called_once_with(config, [{"loss": 0.5}])
    select_checkpoint_and_save_mock.assert_called_once()
    merge_lora_and_save_mock.assert_called_once()
    plot_loss_mock.assert_called_once()
 