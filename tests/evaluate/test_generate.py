import json
import math
import pathlib
 
import pytest
from transformers import AutoTokenizer
import torch
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.evaluate.run import _ensure_output_paths_exist
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
def config(tmp_path) -> Config:
    """Load an evaluate Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()

    test_config.fim_prefix_token = "<|fim_prefix|>"
    test_config.fim_suffix_token = "<|fim_suffix|>"
    test_config.fim_middle_token = "<|fim_middle|>"
    test_config.fim_pad_token = "<|fim_pad|>"
    test_config.eos_token = "<|endoftext|>"
    return test_config 


@pytest.fixture
def tokenizer(mocker):
    """Creates a mock tokenizer instance using nested configurations."""
    tokenizer_instance_mock = mocker.patch("transformers.AutoTokenizer")

    def dummy_convert_tokens_to_ids(token: str) -> int:
        token_to_id_map = {
            "<|fim_pad|>": 0,
            "<|fim_prefix|>": 1,
            "<|fim_suffix|>": 2,
            "<|fim_middle|>": 3,
            "<|endoftext|>": 4
        }
        return token_to_id_map.get(token, -1)

    def dummy_decode(token_ids: list[int], **kwargs) -> str:
        # 14, 15, 16 are the generated tokens from our test model outputs
        if token_ids == [14, 15, 16]:
            return "return a + b;"
        # 10, 11, 12 are distinct from special structural IDs (0-4)
        if token_ids == [10, 11, 12]:
            return "int add(int a, int b) {"
        return ""

    class MockBatchEncoding(dict):
        """Helper to simulate HuggingFace BatchEncoding tensor device placement."""
        def to(self, device):
            return self

    def dummy_pad(features: dict, padding=True, return_tensors=None) -> MockBatchEncoding:
        input_ids = features["input_ids"]
        max_len = max(len(seq) for seq in input_ids)
        
        padded_ids = [seq + [0] * (max_len - len(seq)) for seq in input_ids]
        attention_mask = [[1] * len(seq) + [0] * (max_len - len(seq)) for seq in input_ids]
        
        return MockBatchEncoding({
            "input_ids": torch.tensor(padded_ids),
            "attention_mask": torch.tensor(attention_mask)
        })

    # Setting side effects and static properties directly on the mock
    tokenizer_instance_mock.convert_tokens_to_ids.side_effect = dummy_convert_tokens_to_ids
    tokenizer_instance_mock.decode.side_effect = dummy_decode
    tokenizer_instance_mock.pad.side_effect = dummy_pad
    
    tokenizer_instance_mock.pad_token_id = 0
    tokenizer_instance_mock.pad_token = "<|fim_pad|>"

    return tokenizer_instance_mock

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


# --- _generate_and_save ---

def test_generate_and_save_passes(config, tokenizer, mocker):
    _ensure_output_paths_exist(config)
    # populate benchmark dataset
    with config.benchmark_dataset_path.open("w") as f:
        for example in BENCHMARK_EXAMPLES:
            f.write(json.dumps(example) + "\n")

    lora_model_mock = mocker.MagicMock()
    load_lora_model_mock = mocker.patch("codefinetuner.evaluate.generate._load_lora_model", return_value=lora_model_mock)
    load_tokenizer_mock = mocker.patch("codefinetuner.evaluate.generate._load_tokenizer", return_value=tokenizer)

    generated_middle_token_ids = [14, 15, 16]
    calculated_perplexity = 0.693

    generate_mock = mocker.patch("codefinetuner.evaluate.generate._generate", return_value=[generated_middle_token_ids])
    get_fim_perplexities_mock = mocker.patch("codefinetuner.evaluate.generate._get_fim_perplexities", return_value=[calculated_perplexity])

    config.generation_batch_size = 1

    generate_and_save(config, "checkpoint-test")

    # this must match what tokenizer fixture defines
    generated_middle_tokens = "return a + b;"

    with config.benchmark_evaluation_results_path.open("r") as evaluation_results_file:
        lines = evaluation_results_file.readlines()

    assert len(lines) == len(BENCHMARK_EXAMPLES)

    for line, example in zip(lines, BENCHMARK_EXAMPLES):
        result = json.loads(line)
        assert result["reference_middle"] == example["middle"]
        assert result["base_generated_middle"] == generated_middle_tokens
        assert result["lora_generated_middle"] == generated_middle_tokens
        assert result["base_perplexity"] == calculated_perplexity
        assert result["lora_perplexity"] == calculated_perplexity

    assert tokenizer.decode.call_count == len(BENCHMARK_EXAMPLES) * 2

    expected_calls = len(BENCHMARK_EXAMPLES) * 2
    load_lora_model_mock.assert_called_once()
    load_tokenizer_mock.assert_called_once()
    assert generate_mock.call_count == expected_calls
    assert get_fim_perplexities_mock.call_count == expected_calls
    lora_model_mock.disable_adapter.assert_called()


def test_generate_and_save_raises_runtime_error(config, tokenizer, mocker):
    """Verifies RuntimeError is raised when internal generation fails."""
    _ensure_output_paths_exist(config)
    # populate benchmark dataset
    with config.benchmark_dataset_path.open("w") as f:
        for example in BENCHMARK_EXAMPLES:
            f.write(json.dumps(example) + "\n")

    lora_model_mock = mocker.MagicMock()
    mocker.patch("codefinetuner.evaluate.generate._load_lora_model", return_value=lora_model_mock)
    mocker.patch("codefinetuner.evaluate.generate._load_tokenizer", return_value=tokenizer)

    mocker.patch("codefinetuner.evaluate.generate._generate", side_effect=Exception("Simulated failure"))
    
    config.generation_batch_size = 1

    with pytest.raises(RuntimeError, match="Generation failed at example 1: Simulated failure"):
        generate_and_save(config, "checkpoint-test")
        