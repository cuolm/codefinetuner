import pathlib
import json
import pytest
import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config
from codefinetuner.finetune.train import (
    FIMDataCollator,
    train_lora_model,
    merge_lora_and_save,
    save_log,
    plot_loss
)


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML and redirect paths to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


@pytest.fixture
def tokenizer(config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    return tokenizer


@pytest.fixture
def log_history() -> list:
    return [
        {"loss": 1.254, "step": 10, "epoch": 0.2, "learning_rate": 1e-4},
        {"loss": 0.892, "step": 20, "epoch": 0.4, "learning_rate": 9e-5},
        {"eval_loss": 0.752, "step": 25, "epoch": 0.5},
        {"loss": 0.645, "step": 30, "epoch": 0.6, "learning_rate": 8e-5},
        {"loss": 0.421, "step": 40, "epoch": 0.8, "learning_rate": 7e-5},
        {"loss": 0.312, "step": 50, "epoch": 1.0, "learning_rate": 6e-5},
        {"eval_loss": 0.358, "step": 50, "epoch": 1.0},
    ]


# --- FIMDataCollator ---
 
def test_fim_data_collator_returns_three_keys(tokenizer):
    collator = FIMDataCollator(tokenizer, label_pad_token_id=-100)
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5],    "attention_mask": [1, 1],    "labels": [4, 5]},
    ]
    batch = collator(examples)
    assert set(batch.keys()) == {"input_ids", "attention_mask", "labels"}
 
 
def test_fim_data_collator_output_is_tensors(tokenizer):
    collator = FIMDataCollator(tokenizer, label_pad_token_id=-100)
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
    ]
    batch = collator(examples)
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
 
 
def test_fim_data_collator_pads_to_max_length(tokenizer):
    collator = FIMDataCollator(tokenizer, label_pad_token_id=-100)
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5],    "attention_mask": [1, 1],    "labels": [4, 5]},
    ]
    batch = collator(examples)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)

 
def test_fim_data_collator_pads_input_ids_with_pad_token(tokenizer):
    collator = FIMDataCollator(tokenizer, label_pad_token_id=-100)
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5],    "attention_mask": [1, 1],    "labels": [4, 5]},
    ]
    batch = collator(examples)
    # second example is shorter, its last position should be pad_token_id
    assert batch["input_ids"][1, 2].item() == tokenizer.pad_token_id
 
 
def test_fim_data_collator_pads_attention_mask_with_zeros(tokenizer):
    collator = FIMDataCollator(tokenizer, label_pad_token_id=-100)
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5],    "attention_mask": [1, 1],    "labels": [4, 5]},
    ]
    batch = collator(examples)
    assert batch["attention_mask"][1, 2].item() == 0
 
 
def test_fim_data_collator_pads_labels_with_label_pad_token(tokenizer):
    label_pad = -100
    collator = FIMDataCollator(tokenizer, label_pad_token_id=label_pad)
    examples = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [1, 2, 3]},
        {"input_ids": [4, 5],    "attention_mask": [1, 1],    "labels": [4, 5]},
    ]
    batch = collator(examples)
    assert batch["labels"][1, 2].item() == label_pad
 
 
def test_fim_data_collator_single_example_no_padding_needed(tokenizer):
    collator = FIMDataCollator(tokenizer, label_pad_token_id=-100)
    examples = [{"input_ids": [7, 8], "attention_mask": [1, 1], "labels": [7, 8]}]
    batch = collator(examples)
    assert batch["input_ids"].tolist() == [[7, 8]]


# --- train_lora_model ---

def test_train_lora_model(config, tokenizer, mocker):
    training_arguments_mock = mocker.patch("codefinetuner.finetune.train.TrainingArguments")
    data_collator_mock = mocker.patch("codefinetuner.finetune.train.FIMDataCollator")
    trainer_mock = mocker.patch("codefinetuner.finetune.train.Trainer")

    lora_model_mock = mocker.Mock()
    train_dataset_mock = mocker.Mock()
    eval_dataset_mock= mocker.Mock()

    log_history = train_lora_model(
        config=config,
        lora_model=lora_model_mock,
        tokenizer=tokenizer,
        train_dataset=train_dataset_mock,
        eval_dataset=eval_dataset_mock
    )

    training_arguments_mock.assert_called_once()
    data_collator_mock.assert_called_once()
    trainer_mock.assert_called_once()


# --- merge_lora_and_save ---

def test_merge_lora_and_save(config, tokenizer, mocker):
    auto_model_mock = mocker.patch("codefinetuner.finetune.train.AutoModelForCausalLM.from_pretrained")
    peft_model_mock = mocker.patch("codefinetuner.finetune.train.PeftModel.from_pretrained")
    mocker.patch("codefinetuner.finetune.run.shutil.rmtree")  # mock cleanup to avoid IO errors

    merge_lora_and_save(config, tokenizer)

    auto_model_mock.assert_called_once()
    peft_model_mock.assert_called_once()


# --- save_log ---

def test_save_log_path_exists(config, log_history, tmp_path):
    config.trainer_log_path = tmp_path / "train_log.json"
    save_log(config, log_history)
    assert config.trainer_log_path.exists()


def test_save_log_writes_valid_json(config, log_history, tmp_path):
    config.trainer_log_path = tmp_path / "train_log.json"
    save_log(config, log_history)
    with config.trainer_log_path.open("r", encoding="utf-8") as log_file:
        data = json.load(log_file)
    assert "train" in data
    assert "eval" in data
 
 
def test_save_log_all_keys_present(config, log_history, tmp_path):
    config.trainer_log_path = tmp_path / "train_log.json"
    save_log(config, log_history)
    with config.trainer_log_path.open("r", encoding="utf-8") as log_file:
        data = json.load(log_file)
    assert "steps" in data["train"]
    assert "loss" in data["train"]
    assert "learning_rate" in data["train"]
    assert "epoch" in data["train"]
    assert "steps" in data["eval"]
    assert "loss" in data["eval"]
 

def test_save_log_correct_number_of_entries(config, log_history, tmp_path):
    config.trainer_log_path = tmp_path / "train_log.json"
    save_log(config, log_history)
    with config.trainer_log_path.open("r", encoding="utf-8") as log_file:
        data = json.load(log_file)
    assert len(data["train"]["loss"]) == 5 
    assert len(data["eval"]["loss"]) == 2


# --- plot_loss ---
 
def test_plot_loss(config, tmp_path):
    config.trainer_plot_path = tmp_path / "test_plot.png"
    plot_loss(config)
    assert config.trainer_plot_path.exists()
    assert config.trainer_plot_path.stat().st_size > 0
 