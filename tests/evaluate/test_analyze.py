
import json
import pathlib
 
import numpy as np
import pytest
 
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.analyze import (
    analyze_metric,
    plot_metric_and_save,
    plot_all_metric_averages_and_save,
    save_all_metric_stats,
)
 
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
 
# --- Fixtures ---
 
@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    return Config.load_from_yaml(test_config_path)


@pytest.fixture
def test_evaluation_results():
    return[
        {
            "example_id": 0,
            "reference_middle": "def add(a, b): return a + b",
            "base_generated_middle": "def sum(a, b): return a + b",
            "lora_generated_middle": "def add(a, b): return a + b",
            "base_sentencebleu": 0.3, "lora_sentencebleu": 0.6,
            "base_codebleu": 0.4,     "lora_codebleu": 0.7,     "codebleu_valid": True,
            "base_exact_match": 0.0,  "lora_exact_match": 1.0,
            "base_line_match": 0.0,   "lora_line_match": 1.0,
            "base_perplexity": 6.0,  "lora_perplexity": 3.0,
        },
        {
            "example_id": 1,
            "reference_middle": "def sub(a, b): return a - b",
            "base_generated_middle": "def subtract(a, b): return a - b",
            "lora_generated_middle": "def sub(a, b): return a - b",
            "base_sentencebleu": 0.5, "lora_sentencebleu": 0.8,
            "base_codebleu": 0.6,     "lora_codebleu": 0.9,     "codebleu_valid": True,
            "base_exact_match": 1.0,  "lora_exact_match": 1.0,
            "base_line_match": 1.0,   "lora_line_match": 1.0,
            "base_perplexity": 4.0,   "lora_perplexity": 1.0,
        },
    ]


@pytest.fixture
def test_metric_stats_np():
    return [
        {
            "metric": "sentencebleu",
            "base_array_np": np.array([0.1, 0.9]), 
            "lora_array_np": np.array([0.4, 1.0]),
            "base_average_np": np.float64(0.4),
            "lora_average_np": np.float64(0.7),
            "is_binary": False,
            "higher_is_better": True,
        }, 
        {
            "metric": "exact_match",
            "base_array_np": np.array([0.0, 1.0]),
            "lora_array_np": np.array([1.0, 1.0]),
            "base_average_np": np.float64(0.5),
            "lora_average_np": np.float64(1.0),
            "is_binary": True,
            "higher_is_better": True,
        }
    ]


# --- analyze_metric ---

def test_analyze_metric_correct_averages(config, tmp_path, test_evaluation_results):
    tmp_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"
    config.benchmark_evaluation_results_path = tmp_evaluation_results_path 
    
    with tmp_evaluation_results_path.open("w") as evaluation_results_file:
        for example in test_evaluation_results:
            json_line = json.dumps(example)
            evaluation_results_file.write(json_line + "\n")     

    result = analyze_metric(config, config.codebleu_metric_name, higher_is_better=True) 
    for key in ("metric", "base_array_np", "lora_array_np", "base_average_np",
                "lora_average_np", "is_binary", "higher_is_better"):
        assert key in result
    
    assert result["base_average_np"] == 0.5
    assert result["lora_average_np"] == 0.8
    assert result["higher_is_better"] == True


def test_analyze_metric_exact_and_line_match_are_binary(config, tmp_path, test_evaluation_results):
    tmp_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"
    config.benchmark_evaluation_results_path = tmp_evaluation_results_path 
    
    with tmp_evaluation_results_path.open("w") as evaluation_results_file:
        for example in test_evaluation_results:
            json_line = json.dumps(example)
            evaluation_results_file.write(json_line + "\n")    

    result_exact_match = analyze_metric(config, config.exact_match_metric_name, higher_is_better=True)
    result_line_match = analyze_metric(config, config.line_match_metric_name, higher_is_better=True)
 
    assert result_exact_match["is_binary"] == True
    assert result_line_match["is_binary"] == True
 
 
def test_analyze_metric_sentencebleu_codebleu_perplexity_are_not_binary(config, tmp_path, test_evaluation_results):
    tmp_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"
    config.benchmark_evaluation_results_path = tmp_evaluation_results_path 
    
    with tmp_evaluation_results_path.open("w") as evaluation_results_file:
        for example in test_evaluation_results:
            json_line = json.dumps(example)
            evaluation_results_file.write(json_line + "\n")     
 
    result_sentencebleu = analyze_metric(config, config.sentencebleu_metric_name, higher_is_better=True)
    result_codebleu = analyze_metric(config, config.codebleu_metric_name, higher_is_better=True)
    result_perplexity = analyze_metric(config, config.perplexity_name, higher_is_better=False)
 
    assert result_sentencebleu["is_binary"] == False
    assert result_codebleu["is_binary"] == False
    assert result_perplexity["is_binary"] == False 
 
 
def test_analyze_metric_codebleu_skips_invalid_entries(config, tmp_path, test_evaluation_results):
    """Entries where codebleu_valid is False must be excluded from the score arrays."""
    codebleu_results = [
        {"base_codebleu": 0.9, "lora_codebleu": 0.95, "codebleu_valid": True},
        {"base_codebleu": 0.0, "lora_codebleu": 0.0,  "codebleu_valid": False},
    ]
    tmp_evaluation_results_path = tmp_path / "test_evaluation_results.jsonl"
    config.benchmark_evaluation_results_path = tmp_evaluation_results_path
    with tmp_evaluation_results_path.open("w") as evaluation_results_file:
        for example in codebleu_results :
            json_line = json.dumps(example)
            evaluation_results_file.write(json_line + "\n")    
 
    result = analyze_metric(config, config.codebleu_metric_name, higher_is_better=True)
 
    assert len(result["base_array_np"]) == 1


# --- plot_metric_and_save ---

def test_plot_metric_and_save_continuous_metric(tmp_path, test_metric_stats_np):
    continuous_metric_stats_np = test_metric_stats_np[0]
    tmp_plot_path = tmp_path / "test_plot.png"

    plot_metric_and_save(continuous_metric_stats_np, "sentencebleu", tmp_plot_path)

    assert tmp_plot_path.exists()
    assert tmp_plot_path.stat().st_size > 0  # assert plot file not empty


def test_plot_metric_and_save_binary_metric(tmp_path, test_metric_stats_np):
    binary_metric_stats_np = test_metric_stats_np[1]
    tmp_plot_path = tmp_path / "test_plot.png"

    plot_metric_and_save(binary_metric_stats_np, "exact_match", tmp_plot_path)

    assert tmp_plot_path.exists()
    assert tmp_plot_path.stat().st_size > 0  # assert plot file not empty


# --- save_all_metric_stats ---

def test_save_all_metric_stats(config, tmp_path, test_metric_stats_np):
    tmp_benchmark_analysis_results_path = tmp_path / "test_analysis_results.json"
    config.benchmark_analysis_results_path = tmp_benchmark_analysis_results_path

    save_all_metric_stats(config, test_metric_stats_np)

    assert tmp_benchmark_analysis_results_path.exists()
    assert tmp_benchmark_analysis_results_path.stat().st_size > 0


# --- plot_all_metric_averages_and_save ---

def test_plot_all_metric_averages_and_save(tmp_path, test_metric_stats_np):
    tmp_plot_path = tmp_path / "all_metric_averages_plot.png"

    plot_all_metric_averages_and_save(test_metric_stats_np, tmp_plot_path)

    assert tmp_plot_path.exists()
    assert tmp_plot_path.stat().st_size > 0

