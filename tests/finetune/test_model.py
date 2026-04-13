import pathlib
import json
import pytest
import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config
from codefinetuner.finetune.model import (
    load_and_configure_lora_model
)


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML and redirect paths to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- load_and_configure_lora_model ---

def test_load_and_configure_lora_model_cuda(config, mocker):
    # force CUDA path to test quantization and kbit preparation logic
    config.device = "cuda"
    
    bnb_mock = mocker.patch("codefinetuner.finetune.model.BitsAndBytesConfig")
    auto_model_mock = mocker.patch("codefinetuner.finetune.model.AutoModelForCausalLM.from_pretrained")
    kbit_prep_mock = mocker.patch("codefinetuner.finetune.model.prepare_model_for_kbit_training")
    lora_cfg_mock = mocker.patch("codefinetuner.finetune.model.LoraConfig")
    get_peft_mock = mocker.patch("codefinetuner.finetune.model.get_peft_model")

    load_and_configure_lora_model(config)

    bnb_mock.assert_called_once()
    auto_model_mock.assert_called_once()
    kbit_prep_mock.assert_called_once()
    lora_cfg_mock.assert_called_once()
    get_peft_mock.assert_called_once()


def test_load_and_configure_lora_model_mps(config, mocker):
    config.device = "mps"
    auto_model_mock = mocker.patch("codefinetuner.finetune.model.AutoModelForCausalLM.from_pretrained")
    lora_cfg_mock = mocker.patch("codefinetuner.finetune.model.LoraConfig")
    get_peft_mock = mocker.patch("codefinetuner.finetune.model.get_peft_model")
    
    load_and_configure_lora_model(config)

    auto_model_mock.assert_called_once()
    auto_model_mock.return_value.to.assert_called_once_with("mps")
    lora_cfg_mock.assert_called_once()
    get_peft_mock.assert_called_once()


def test_load_and_configure_lora_model_cpu(config, mocker):
    config.device = "cpu"
    auto_model_mock = mocker.patch("codefinetuner.finetune.model.AutoModelForCausalLM.from_pretrained")
    lora_cfg_mock = mocker.patch("codefinetuner.finetune.model.LoraConfig")
    get_peft_mock = mocker.patch("codefinetuner.finetune.model.get_peft_model")
    
    load_and_configure_lora_model(config)

    auto_model_mock.assert_called_once()
    auto_model_mock.return_value.to.assert_called_once_with("cpu")
    lora_cfg_mock.assert_called_once()
    get_peft_mock.assert_called_once()