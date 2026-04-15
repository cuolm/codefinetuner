
import pathlib
import textwrap

import pytest
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.evaluate.config import Config
 
 
# --- Fixtures ---
 
@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    return test_config


# --- Config.load_from_yaml ---

def test_load_from_yaml_success(config):
    assert config.model_name == "tests/models/Qwen2.5-Coder-0.5B"
    assert config.fim_pad_token == "<|fim_pad|>"


def test_load_from_yaml_missing_file(tmp_path):
    nonexistent_yaml = tmp_path / "nonexistent_yaml.yaml"
    with pytest.raises(FileNotFoundError):
        Config.load_from_yaml(nonexistent_yaml)


def test_load_from_yaml_invalid_yaml(tmp_path):
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("finetune:\n  key: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to load YAML"):
        Config.load_from_yaml(invalid_yaml)


def test_load_from_yaml_ignores_unknown_keys(tmp_path):
    """Extra YAML keys (e.g. from global anchors) must not raise."""
    config_text = textwrap.dedent("""
        evaluate:
          workspace_path: "tests"
          model_name: "Qwen/Qwen2.5-Coder-1.5B"
          fim_prefix_token: "<|fim_prefix|>"
          fim_middle_token: "<|fim_middle|>"
          fim_suffix_token: "<|fim_suffix|>"
          fim_pad_token: "<|fim_pad|>"
          eos_token: "<|endoftext|>"
          label_pad_token_id: -100
          data_language: "c"
          data_extensions: [".c", ".h"]
          unknown_key_that_does_not_exist: 999
    """)
    test_config_path = tmp_path / "test_config.yaml"
    test_config_path.write_text(config_text, encoding="utf-8")
    test_config = Config.load_from_yaml(test_config_path)
    assert test_config.model_name == "Qwen/Qwen2.5-Coder-1.5B"


# --- _validate_metric_weights ---
 
def test_validate_metric_weights_raises_when_codebleu_weights_do_not_sum_to_one(config):
    config.codebleu_ngram_weight = 0.5
    config.codebleu_weighted_ngram_weight = 0.5
    config.codebleu_syntax_ast_weight = 0.5
    config.codebleu_dataflow_weight = 0.5
    with pytest.raises(ValueError, match="CodeBLEU"):
        config._validate_metric_weights()
 
 
def test_validate_metric_weights_raises_when_sentencebleu_weights_do_not_sum_to_one(config):
    config.sentencebleu_ngram_weight_1 = 0.5
    config.sentencebleu_ngram_weight_2 = 0.5
    config.sentencebleu_ngram_weight_3 = 0.5
    config.sentencebleu_ngram_weight_4 = 0.5
    with pytest.raises(ValueError, match="SentenceBLEU"):
        config._validate_metric_weights()


# --- _setup_paths ---

def test_setup_paths_dataset_paths_are_under_workspace(config):
    assert str(config.benchmark_dataset_path).startswith(str(config.workspace_path))
    assert str(config.benchmark_evaluation_results_path).startswith(str(config.workspace_path))
    assert str(config.benchmark_analysis_results_path).startswith(str(config.workspace_path))


# --- _ensure_output_paths_exist ---

def test_ensure_output_paths_exist_creates_parent_dirs(config):
    assert config.evaluate_outputs_dir_path.parent.exists()
    assert config.benchmark_dataset_path.parent.exists()
    assert config.benchmark_evaluation_results_dir.parent.exists()
    assert config.benchmark_evaluation_results_path.parent.exists() 
    assert config.benchmark_analysis_results_path.parent.exists()


# --- ensure_nltk_initialized ---
 
def test_ensure_nltk_initialized_sets_flag_to_true(config):
    config._nltk_initialized = False
    config.ensure_nltk_initialized()
    assert config._nltk_initialized is True
