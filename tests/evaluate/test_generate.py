import json
import math
import pathlib
 
import pytest
from transformers import AutoTokenizer
import torch
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.generate import (
    _load_lora_model,
    _get_fim_perplexity,
    _generate,
    generate_and_save,
)
 
 
# --- Fixtures ---
 
@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    return Config.load_from_yaml(test_config_path)


@pytest.fixture
def tokenizer(config) -> AutoTokenizer:
    """Load the pinned local tokenizer from tests/models/."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer


# --- _load_lora_model ---

def test_load_lora_model(config, mocker, tmp_path):
    base_model = mocker.Mock()
    base_model_mock = mocker.patch("codefinetuner.evaluate.generate.AutoModelForCausalLM.from_pretrained", return_value=base_model)
    lora_model = mocker.Mock()
    peft_model_mock = mocker.patch("codefinetuner.evaluate.generate.PeftModel.from_pretrained", return_value = lora_model)
    lora_model.to.return_value = lora_model

    result = _load_lora_model(config, tmp_path / "checkpoint-test")

    base_model_mock.assert_called_once_with(
        pretrained_model_name_or_path=config.model_name,
        dtype=config.model_dtype,
        low_cpu_mem_usage=True,
    )
    peft_model_mock.assert_called_once_with(
        model=base_model,
        model_id=tmp_path / "checkpoint-test",
    )
    lora_model.to.assert_called_once_with(config.device)
    lora_model.eval.assert_called_once()

    assert result is lora_model


# --- _get_fim_perplexity ---

def test_get_fim_perplexity_calculation_passes(config, mocker):
    model_mock = mocker.Mock()
    model_mock.return_value.loss = torch.tensor(0.693)

    input_ids = [1, 2, 3]
    label_ids = [1, 2, 3]

    calculated_perplexity = _get_fim_perplexity(config, model_mock, input_ids, label_ids)

    model_mock.assert_called_once()

    expected_perplexity = math.exp(0.693)
    assert calculated_perplexity == pytest.approx(expected_perplexity)


def test_get_fim_perplexity_calculation_raises_error(config, mocker):
    model_mock = mocker.Mock()
    model_mock.side_effect = ValueError("Simulated failure")

    input_ids = [1, 2, 3]
    label_ids = [1, 2, 3]

    calculated_perplexity = _get_fim_perplexity(config, model_mock, input_ids, label_ids)

    assert calculated_perplexity == float("inf")


# --- _generate ---

def test_generate(config, tokenizer, mocker):
    model_mock = mocker.Mock()
    generated_token_ids_tensor = torch.tensor([[1, 2, 3, 4, 5, 6]])
    model_mock.generate.return_value = generated_token_ids_tensor
    prompt_token_ids = [1, 2, 3]
    
    generated_middle_token_ids = _generate(config, model_mock, tokenizer, prompt_token_ids)
    assert generated_middle_token_ids == [4, 5, 6] 


# --- generate_and_save ---

def test_generate_and_save_passes(config, tokenizer, mocker, tmp_path):
    """ uses tests/outputs/evaluate/datasets/test_benchmark_dataset.jsonl """
    lora_model_mock = mocker.MagicMock()
    load_lora_model_mock = mocker.patch("codefinetuner.evaluate.generate._load_lora_model", return_value=lora_model_mock)
    load_tokenizer_mock = mocker.patch("codefinetuner.evaluate.generate._load_tokenizer", return_value=tokenizer)

    calculated_perplexity = 0.693
    get_fim_perplexity_mock = mocker.patch("codefinetuner.evaluate.generate._get_fim_perplexity", return_value=calculated_perplexity)

    generated_middle_token_ids = [4, 5, 6]
    generate_mock = mocker.patch("codefinetuner.evaluate.generate._generate", return_value=generated_middle_token_ids)
    
    config.benchmark_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"

    generate_and_save(config, "checkpoint-test")

    generated_middle_tokens = tokenizer.decode(generated_middle_token_ids, skip_special_tokens=True)
    
    with config.benchmark_evaluation_results_path.open("r") as evaluation_results_file:
        lines = evaluation_results_file.readlines()
        assert len(lines) > 0 
        for line in lines:
            result = json.loads(line)
            assert result["base_generated_middle"] == generated_middle_tokens
            assert result["lora_generated_middle"] == generated_middle_tokens
            assert result["base_perplexity"] == calculated_perplexity
            assert result["lora_perplexity"] == calculated_perplexity
    
    dataset_length = len(lines)
    expected_calls = dataset_length * 2 
    
    load_lora_model_mock.assert_called_once()
    load_tokenizer_mock.assert_called_once()
    assert generate_mock.call_count == expected_calls
    assert get_fim_perplexity_mock.call_count == expected_calls
    lora_model_mock.disable_adapter.assert_called()


def test_generate_and_save_raises_runtime_error(config, tokenizer, mocker, tmp_path):
    """ verifies RuntimeError is raised when internal generation fails """
    lora_model_mock = mocker.MagicMock()
    mocker.patch("codefinetuner.evaluate.generate._load_lora_model", return_value=lora_model_mock)
    mocker.patch("codefinetuner.evaluate.generate._load_tokenizer", return_value=tokenizer)

    mocker.patch("codefinetuner.evaluate.generate._generate", side_effect=Exception("Simulated failure"))
    
    config.benchmark_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"

    with pytest.raises(RuntimeError, match="Generation failed at example 0: Simulated failure"):
        generate_and_save(config, "checkpoint-test")
    
     