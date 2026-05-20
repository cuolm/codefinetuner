import json
import pathlib
import pytest

from codefinetuner.evaluate.run import _ensure_output_paths_exist
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.evaluate import evaluate_and_save

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load an evaluate Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config 


@pytest.fixture
def test_evaluation_results():
    return {
        "example_id": 0,
        "reference_middle": "def add(a, b): return a + b",
        "base_generated_middle": "def sum(a, b): return a + b",
        "lora_generated_middle": "def add(a, b): return a + b",
        "base_perplexity": 1.5,
        "lora_perplexity": 1.2
    }


# --- evaluate_and_save ---

def test_evaluate_and_save_passes(config, mocker, test_evaluation_results):
    _ensure_output_paths_exist(config)
    with config.benchmark_evaluation_results_path.open("w") as evaluation_results_file:
        evaluation_results_file.write(json.dumps(test_evaluation_results) + "\n")

    get_codebleu_mock = mocker.patch("codefinetuner.evaluate.evaluate.get_codebleu", return_value=(0.9, True))
    get_sentencebleu_mock = mocker.patch("codefinetuner.evaluate.evaluate.get_sentencebleu", return_value=0.8)
    get_exact_match_mock = mocker.patch("codefinetuner.evaluate.evaluate.get_exact_match", return_value=True)
    get_line_match_mock = mocker.patch("codefinetuner.evaluate.evaluate.get_line_match", return_value=1.0)

    evaluate_and_save(config)

    # after update of the evaluation_results_file file with evaluation results such as codebleu...
    with config.benchmark_evaluation_results_path.open("r") as evaluation_results_file:
        lines = evaluation_results_file.readlines()
        assert len(lines) == 1
        result = json.loads(lines[0])
        
        assert result["example_id"] == test_evaluation_results["example_id"]
        assert result["reference_middle"] == test_evaluation_results["reference_middle"]
        assert result["base_generated_middle"] == test_evaluation_results["base_generated_middle"]
        assert result["lora_generated_middle"] == test_evaluation_results["lora_generated_middle"]
        assert result["base_perplexity"] == test_evaluation_results["base_perplexity"]
        assert result["lora_perplexity"] == test_evaluation_results["lora_perplexity"]
        
        assert result["base_codebleu"] == 0.9
        assert result["codebleu_valid"] is True
        assert result["base_sentencebleu"] == 0.8
        assert result["base_exact_match"] is True
        assert result["base_line_match"] == 1.0

    assert get_codebleu_mock.call_count == 2
    assert get_sentencebleu_mock.call_count == 2
    assert get_exact_match_mock.call_count == 2
    assert get_line_match_mock.call_count == 2


def test_evaluate_and_save_raises_runtime_error(config, mocker, test_evaluation_results):
    _ensure_output_paths_exist(config)
    with config.benchmark_evaluation_results_path.open("w") as evaluation_results_file:
        evaluation_results_file.write(json.dumps(test_evaluation_results) + "\n")

    mocker.patch("codefinetuner.evaluate.evaluate.get_codebleu", side_effect=Exception("Simulated metric failure"))
    temp_path = config.benchmark_evaluation_results_path.with_name(f"{config.benchmark_evaluation_results_path.name}.tmp")

    with pytest.raises(RuntimeError, match="Evaluation failed at example 0: Simulated metric failure"):
        evaluate_and_save(config)

    # test clean up of temp_path on error
    assert not temp_path.exists()
