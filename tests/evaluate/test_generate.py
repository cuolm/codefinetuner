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
    _load_hf_lora_model,
    _load_unsloth_lora_model,
    _get_fim_perplexities,
    _generate,
    generate_and_save,
)


BENCHMARK_EXAMPLES = [
    {
        "example_token_ids": [1, 2, 3, 4, 5],
        "prefix_token_ids": [1, 2],
        "suffix_token_ids": [4, 5],
        "middle_token_ids": [3],
        "prefix": "int add(int a, int b) {",
        "suffix": "}",
        "middle": "return a + b;"
    },
    {
        "example_token_ids": [6, 7, 8, 9, 10],
        "prefix_token_ids": [6, 7],
        "suffix_token_ids": [9, 10],
        "middle_token_ids": [8],
        "prefix": "int subtract(int a, int b) {",
        "suffix": "}",
        "middle": "return a - b;"
    }
]
 
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


@pytest.fixture
def benchmark_dataset_path(tmp_path):
    path = tmp_path / "test_benchmark_dataset.jsonl"
    with path.open("w") as f:
        for example in BENCHMARK_EXAMPLES:
            f.write(json.dumps(example) + "\n")
    return path


# --- _load_lora_model ---

def test_load_lora_model_dispatches_to_hf(config, mocker, tmp_path):
    config.use_unsloth = False

    hf_mock = mocker.patch("codefinetuner.evaluate.generate._load_hf_lora_model")
    _load_lora_model(config, tmp_path / "checkpoint-test")

    hf_mock.assert_called_once_with(config, tmp_path / "checkpoint-test")


def test_load_lora_model_dispatches_to_unsloth(config, mocker, tmp_path):
    config.use_unsloth = True

    unsloth_mock = mocker.patch("codefinetuner.evaluate.generate._load_unsloth_lora_model")
    _load_lora_model(config, tmp_path / "checkpoint-test")

    unsloth_mock.assert_called_once_with(config, tmp_path / "checkpoint-test")


# --- _load_hf_lora_model ---

def test_load_hf_lora_model(config, mocker, tmp_path):
    config.use_unsloth = False

    base_model = mocker.Mock()
    base_model_mock = mocker.patch("codefinetuner.evaluate.generate.AutoModelForCausalLM.from_pretrained", return_value=base_model)
    lora_model = mocker.Mock()
    peft_model_mock = mocker.patch("codefinetuner.evaluate.generate.PeftModel.from_pretrained", return_value=lora_model)
    lora_model.to.return_value = lora_model

    result = _load_hf_lora_model(config, tmp_path / "checkpoint-test")

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


# --- _load_unsloth_lora_model ---

def test_load_unsloth_lora_model(config, mocker, tmp_path):
    config.use_unsloth = True

    mock_unsloth = mocker.MagicMock()
    mocker.patch.dict("sys.modules", {"unsloth": mock_unsloth})

    mock_model = mocker.MagicMock()
    mock_unsloth.FastLanguageModel.from_pretrained.return_value = (mock_model, None)
    lora_model = mocker.Mock()
    peft_model_mock = mocker.patch("codefinetuner.evaluate.generate.PeftModel.from_pretrained", return_value=lora_model)

    result = _load_unsloth_lora_model(config, tmp_path / "checkpoint-test")

    mock_unsloth.FastLanguageModel.from_pretrained.assert_called_once_with(
        model_name=config.model_name,
        max_seq_length=config.max_token_sequence_length,
        dtype=config.model_dtype,
        load_in_4bit=True,
    )
    peft_model_mock.assert_called_once_with(
        model=mock_model,
        model_id=tmp_path / "checkpoint-test",
    )
    lora_model.eval.assert_called_once()
    assert result is lora_model


def test_load_unsloth_lora_model_raises_import_error(config, mocker, tmp_path):
    config.use_unsloth = True

    mocker.patch.dict("sys.modules", {"unsloth": None})

    with pytest.raises(ImportError, match="Unsloth is not installed."):
        _load_unsloth_lora_model(config, tmp_path / "checkpoint-test")



# --- _get_fim_perplexities ---

def test_get_fim_perplexities_calculation_passes(config, mocker):
    model_mock = mocker.Mock()
    model_mock.return_value.loss = torch.tensor(0.693)

    input_ids_batch = [[1, 2, 3], [1, 2, 3]]
    label_ids_batch = [[1, 2, 3], [1, 2, 3]]

    calculated_perplexities = _get_fim_perplexities(config, model_mock, input_ids_batch, label_ids_batch)

    expected_perplexities = [math.exp(0.693), math.exp(0.693)]
    assert calculated_perplexities == pytest.approx(expected_perplexities)


def test_get_fim_perplexities_calculation_raises_error(config, mocker):
    model_mock = mocker.Mock()
    model_mock.side_effect = ValueError("Simulated failure")

    input_ids_batch = [[1, 2, 3], [1, 2, 3]]
    label_ids_batch = [[1, 2, 3], [1, 2, 3]]

    calculated_perplexities = _get_fim_perplexities(config, model_mock, input_ids_batch, label_ids_batch)

    assert calculated_perplexities == [float("inf"), float("inf")]


# --- _generate ---

def test_generate(config, tokenizer, mocker):
    model_mock = mocker.Mock()
    generated_token_ids_tensor = torch.tensor([[1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15, 16]])
    model_mock.generate.return_value = generated_token_ids_tensor
    prompt_token_ids = [[1, 2, 3], [11, 12, 13]]
    
    generated_middle_token_ids_batch = _generate(config, model_mock, tokenizer, prompt_token_ids)
    assert generated_middle_token_ids_batch == [[4, 5, 6], [14, 15, 16]]


# --- generate_and_save ---

def test_generate_and_save_passes(config, tokenizer, mocker, tmp_path):
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




def test_generate_and_save_passes(config, tokenizer, mocker, tmp_path, benchmark_dataset_path):
    num_examples = len(BENCHMARK_EXAMPLES)

    lora_model_mock = mocker.MagicMock()
    load_lora_model_mock = mocker.patch("codefinetuner.evaluate.generate._load_lora_model", return_value=lora_model_mock)
    load_tokenizer_mock = mocker.patch("codefinetuner.evaluate.generate._load_tokenizer", return_value=tokenizer)

    generated_middle_token_ids = [14, 15, 16]
    calculated_perplexity = 0.693

    generate_mock = mocker.patch("codefinetuner.evaluate.generate._generate", return_value=[generated_middle_token_ids])
    get_fim_perplexities_mock = mocker.patch("codefinetuner.evaluate.generate._get_fim_perplexities", return_value=[calculated_perplexity])

    config.generation_batch_size = 1
    config.benchmark_dataset_path = benchmark_dataset_path
    config.benchmark_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"

    generate_and_save(config, "checkpoint-test")

    generated_middle_tokens = tokenizer.decode(generated_middle_token_ids, skip_special_tokens=True)

    with config.benchmark_evaluation_results_path.open("r") as evaluation_results_file:
        lines = evaluation_results_file.readlines()

    assert len(lines) == num_examples

    for line, example in zip(lines, BENCHMARK_EXAMPLES):
        result = json.loads(line)
        assert result["reference_middle"] == example["middle"]
        assert result["base_generated_middle"] == generated_middle_tokens
        assert result["lora_generated_middle"] == generated_middle_tokens
        assert result["base_perplexity"] == calculated_perplexity
        assert result["lora_perplexity"] == calculated_perplexity

    expected_calls = num_examples * 2
    load_lora_model_mock.assert_called_once()
    load_tokenizer_mock.assert_called_once()
    assert generate_mock.call_count == expected_calls
    assert get_fim_perplexities_mock.call_count == expected_calls
    lora_model_mock.disable_adapter.assert_called()


def test_generate_and_save_raises_runtime_error(config, tokenizer, mocker, tmp_path, benchmark_dataset_path):
    """ verifies RuntimeError is raised when internal generation fails """
    lora_model_mock = mocker.MagicMock()
    mocker.patch("codefinetuner.evaluate.generate._load_lora_model", return_value=lora_model_mock)
    mocker.patch("codefinetuner.evaluate.generate._load_tokenizer", return_value=tokenizer)

    mocker.patch("codefinetuner.evaluate.generate._generate", side_effect=Exception("Simulated failure"))
    
    config.generation_batch_size = 1
    config.benchmark_dataset_path = benchmark_dataset_path
    config.benchmark_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"

    with pytest.raises(RuntimeError, match="Generation failed at example 1: Simulated failure"):
        generate_and_save(config, "checkpoint-test")
    
     