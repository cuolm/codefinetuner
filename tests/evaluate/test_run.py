import pathlib
 
import pytest
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"

from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.run import (
    _ensure_output_paths_exist,
    _ensure_nltk_initialized,
    _get_checkpoint_path,
    run,
)
 
 
# --- Fixtures ---
 
@pytest.fixture
def config(tmp_path) -> Config:
    """Load an evaluate Config from the test YAML."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config._setup_paths()
    return test_config 


# --- _ensure_output_paths_exist ---

def test_ensure_output_paths_exist_creates_directories(config):
    # Ensure directories do not exist initially
    assert not config.evaluate_outputs_dir_path.exists()
    assert not config.benchmark_evaluation_results_dir.exists()

    _ensure_output_paths_exist(config)

    # Check that parent directories were correctly created on the file system
    assert config.evaluate_outputs_dir_path.exists()
    assert config.benchmark_evaluation_results_dir.exists()


# --- _ensure_nltk_initialized ---

def test_ensure_nltk_initialized_skips_if_already_done(config, mocker):
    download_mock = mocker.patch("nltk.download")
    config._nltk_initialized = True

    _ensure_nltk_initialized(config)
    download_mock.assert_not_called()


def test_ensure_nltk_initialized_runs_successfully(config, mocker):
    download_mock = mocker.patch("nltk.download")
    config._nltk_initialized = False

    _ensure_nltk_initialized(config)
    assert download_mock.call_count == 2
    assert config._nltk_initialized is True


def test_ensure_nltk_initialized_raises_runtime_error_on_failure(config, mocker):
    mocker.patch("nltk.download", side_effect=Exception("Network connection timeout"))
    config._nltk_initialized = False

    with pytest.raises(RuntimeError, match="NLTK initializerion failed"):
        _ensure_nltk_initialized(config)


# --- _get_checkpoint_path ---

def test_get_checkpoint_path_pipeline(config):
    _ensure_output_paths_exist(config)
    config.generation_checkpoint = "pipeline"
    expected_path = config.finetune_outputs_path / "results" / "selected_checkpoint"
    expected_path.mkdir(parents=True) 
    checkpoint_path = _get_checkpoint_path(config)
    assert checkpoint_path == expected_path


def test_get_checkpoint_path_specific_checkpoint(config):
    _ensure_output_paths_exist(config)
    config.generation_checkpoint = "checkpoint-100"
    expected_path = config.finetune_outputs_path / "checkpoints" / "checkpoint-100"
    expected_path.mkdir(parents=True)
    checkpoint_path = _get_checkpoint_path(config)
    assert checkpoint_path == expected_path


def test_get_checkpoint_path_raises_error_if_missing(config):
    _ensure_output_paths_exist(config)
    config.generation_checkpoint = "missing-checkpoint"
    
    # Missing checkpint dir
    
    with pytest.raises(RuntimeError, match="Checkpoint path not found"):
        _get_checkpoint_path(config)
 

# --- run ---

def test_run_calls_all_internal_functions(config, mocker):
    _ensure_output_paths_exist(config)
    nltk_initialized_mock = mocker.patch("codefinetuner.evaluate.run._ensure_nltk_initialized")
    create_benchmark_dataset_mock = mocker.patch("codefinetuner.evaluate.run.create_benchmark_dataset", return_value=10)
    mocker.patch("pathlib.Path.exists", return_value=True)
    checkpoint_path = pathlib.Path("tmp_finetune_outputs_path/checkpoints/checkpoint-test")
    mocker.patch("codefinetuner.evaluate.run._get_checkpoint_path", return_value=checkpoint_path)
    generate_and_save_mock = mocker.patch("codefinetuner.evaluate.run.generate_and_save")
    evaluate_and_save_mock = mocker.patch("codefinetuner.evaluate.run.evaluate_and_save")
    analyze_metric_mock = mocker.patch(
        "codefinetuner.evaluate.run.analyze_metric",
        return_value={
            "metric": "sentencebleu",
            "base_array_np": [],
            "lora_array_np": [],
            "base_average_np": 0.0,
            "lora_average_np": 0.0,
            "is_binary": False,
            "higher_is_better": True,
        },
    )
    get_plot_path_mock = mocker.patch("codefinetuner.evaluate.run.get_plot_path", return_value=pathlib.Path("/tmp/plot.png"))
    plot_metric_and_save_mock = mocker.patch("codefinetuner.evaluate.run.plot_metric_and_save")
    plot_all_metric_averages_mock = mocker.patch("codefinetuner.evaluate.run.plot_all_metric_averages_and_save")
    save_all_metric_stats_mock = mocker.patch("codefinetuner.evaluate.run.save_all_metric_stats")

    config.plot_only = False
    config.benchmark_use_existing_dataset = False  # forces create_benchmark_dataset
    config.generation_checkpoint = "checkpoint-test"  
    config.finetune_outputs_path = pathlib.Path("tmp_finetune_outputs_path")

    run(config)

    nltk_initialized_mock.assert_called_once_with(config)
    create_benchmark_dataset_mock.assert_called_once_with(config)

    generate_and_save_mock.assert_called_once_with(config, config.finetune_outputs_path / "checkpoints" / "checkpoint-test")
    evaluate_and_save_mock.assert_called_once_with(config)

    assert analyze_metric_mock.call_count == len(config.metric_configs)
    assert plot_metric_and_save_mock.call_count == len(config.metric_configs)
    assert get_plot_path_mock.call_count == len(config.metric_configs) + 1

    plot_all_metric_averages_mock.assert_called_once()
    save_all_metric_stats_mock.assert_called_once()
    