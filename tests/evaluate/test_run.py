import pathlib
 
import pytest
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.run import _ensure_checkpoints, run
 
 
# --- Fixtures ---
 
@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    return Config.load_from_yaml(test_config_path)
 
 
# --- _ensure_checkpoints ---
 
def test_ensure_checkpoints_raises_when_directory_does_not_exist(config, tmp_path):
    config.trainer_checkpoints_dir_path = tmp_path / "nonexistent"
    with pytest.raises(RuntimeError):
        _ensure_checkpoints(config)


def test_ensure_checkpoints_passes_when_checkpoint_exists(config, tmp_path):
    config.trainer_checkpoints_dir_path = tmp_path
    checkpoint_path = tmp_path / "checkpoint-100"
    checkpoint_path.mkdir()
    _ensure_checkpoints(config)


# --- run ---

def test_run_calls_all_internal_functions(config, mocker):
    create_benchmark_dataset_mock = mocker.patch("codefinetuner.evaluate.run.create_benchmark_dataset", return_value=10)
    ensure_checkpoints_mock = mocker.patch("codefinetuner.evaluate.run._ensure_checkpoints")
    get_last_checkpoint_mock = mocker.patch("codefinetuner.evaluate.run.get_last_checkpoint", return_value="checkpoint-test")
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
    config.trainer_checkpoint = "last"  # forces get_last_checkpoint

    run(config)

    create_benchmark_dataset_mock.assert_called_once_with(config)

    ensure_checkpoints_mock.assert_called_once_with(config)
    get_last_checkpoint_mock.assert_called_once_with(config.trainer_checkpoints_dir_path)
    generate_and_save_mock.assert_called_once_with(config, "checkpoint-test")
    evaluate_and_save_mock.assert_called_once_with(config)

    assert analyze_metric_mock.call_count == len(config.metric_configs)
    assert plot_metric_and_save_mock.call_count == len(config.metric_configs)
    assert get_plot_path_mock.call_count == len(config.metric_configs) + 1

    plot_all_metric_averages_mock.assert_called_once()
    save_all_metric_stats_mock.assert_called_once()
    