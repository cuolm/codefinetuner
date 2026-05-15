import pathlib
 
import pytest
 
tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"
 
from codefinetuner.evaluate.config import Config
from codefinetuner.evaluate.run import run
 
 
# --- Fixtures ---
 
@pytest.fixture
def config() -> Config:
    """Load an evaluate Config from the test YAML."""
    return Config.load_from_yaml(test_config_path)
 

# --- run ---

def test_run_calls_all_internal_functions(config, mocker):
    create_benchmark_dataset_mock = mocker.patch("codefinetuner.evaluate.run.create_benchmark_dataset", return_value=10)
    mocker.patch("pathlib.Path.exists", return_value=True)
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

    create_benchmark_dataset_mock.assert_called_once_with(config)

    generate_and_save_mock.assert_called_once_with(config, config.finetune_outputs_path / "checkpoints" / "checkpoint-test")
    evaluate_and_save_mock.assert_called_once_with(config)

    assert analyze_metric_mock.call_count == len(config.metric_configs)
    assert plot_metric_and_save_mock.call_count == len(config.metric_configs)
    assert get_plot_path_mock.call_count == len(config.metric_configs) + 1

    plot_all_metric_averages_mock.assert_called_once()
    save_all_metric_stats_mock.assert_called_once()
    