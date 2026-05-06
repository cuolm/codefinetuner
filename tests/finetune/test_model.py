import pathlib
import pytest
from datasets import IterableDataset
from transformers import AutoTokenizer

tests_path = pathlib.Path(__file__).parent.parent 
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.finetune.config import Config
from codefinetuner.finetune.model import (
    load_and_configure_lora_model,
    _load_hf_model,
    _load_unsloth_model
)


# --- Fixtures ---

@pytest.fixture
def config() -> Config:
    """Load a Config from the test YAML and redirect paths to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- load_and_configure_lora_model ---

def test_load_and_configure_lora_model_dispatches_to_hf(config, mocker):
    config.use_unsloth = False

    hf_mock = mocker.patch("codefinetuner.finetune.model._load_hf_model")
    load_and_configure_lora_model(config)

    hf_mock.assert_called_once_with(config)


def test_load_and_configure_lora_model_dispatches_to_unsloth(config, mocker):
    config.use_unsloth = True

    unsloth_mock = mocker.patch("codefinetuner.finetune.model._load_unsloth_model")
    load_and_configure_lora_model(config)

    unsloth_mock.assert_called_once_with(config)


# --- _load_hf_model ---

def test_load_hf_model_cuda(config, mocker):
    config.device = "cuda"
    config.use_unsloth = False

    bnb_mock = mocker.patch("codefinetuner.finetune.model.BitsAndBytesConfig")
    auto_model_mock = mocker.patch("codefinetuner.finetune.model.AutoModelForCausalLM.from_pretrained")
    kbit_prep_mock = mocker.patch("codefinetuner.finetune.model.prepare_model_for_kbit_training")
    lora_cfg_mock = mocker.patch("codefinetuner.finetune.model.LoraConfig")
    get_peft_mock = mocker.patch("codefinetuner.finetune.model.get_peft_model")

    _load_hf_model(config)

    bnb_mock.assert_called_once()
    auto_model_mock.assert_called_once()
    kbit_prep_mock.assert_called_once()
    lora_cfg_mock.assert_called_once()
    get_peft_mock.assert_called_once()


def test_load_hf_model_mps(config, mocker):
    config.device = "mps"
    config.use_unsloth = False

    auto_model_mock = mocker.patch("codefinetuner.finetune.model.AutoModelForCausalLM.from_pretrained")
    lora_cfg_mock = mocker.patch("codefinetuner.finetune.model.LoraConfig")
    get_peft_mock = mocker.patch("codefinetuner.finetune.model.get_peft_model")
    
    _load_hf_model(config)

    auto_model_mock.assert_called_once()
    auto_model_mock.return_value.to.assert_called_once_with("mps")
    lora_cfg_mock.assert_called_once()
    get_peft_mock.assert_called_once()


def test_load_hf_model_cpu(config, mocker):
    config.device = "cpu"
    config.use_unsloth = False

    auto_model_mock = mocker.patch("codefinetuner.finetune.model.AutoModelForCausalLM.from_pretrained")
    lora_cfg_mock = mocker.patch("codefinetuner.finetune.model.LoraConfig")
    get_peft_mock = mocker.patch("codefinetuner.finetune.model.get_peft_model")
    
    _load_hf_model(config)

    auto_model_mock.assert_called_once()
    auto_model_mock.return_value.to.assert_called_once_with("cpu")
    lora_cfg_mock.assert_called_once()
    get_peft_mock.assert_called_once()


# --- load_unsloth_model ---

def test_load_unsloth_model(config, mocker):
    config.device = "cuda"
    config.use_unsloth = True

    # Mock the unsloth module in sys.modules
    mock_unsloth = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"unsloth": mock_unsloth})
    
    # FastLanguageModel.from_pretrained returns a tuple: (model, tokenizer)
    mock_model = mocker.MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained.return_value = (mock_model, None)

    load_and_configure_lora_model(config)

    mock_unsloth.FastLanguageModel.from_pretrained.assert_called_once()
    mock_unsloth.FastLanguageModel.get_peft_model.assert_called_once()


def test_load_unsloth_model_non_cuda(config, mocker):
    config.device = "cuda"
    config.use_unsloth = True

    mocker.patch.dict("sys.modules", {"unsloth": None})

    with pytest.raises(ImportError, match="Unsloth is not installed"):
        load_and_configure_lora_model(config)
