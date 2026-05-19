import json
import pathlib

import numpy as np
import pytest

from codefinetuner.preprocess.run import _ensure_output_paths_exist
from codefinetuner.preprocess.config import Config
from codefinetuner.preprocess.analyze import (
    _extract_middle_length,
    _load_dataset_stats,
    _plot_middle_distribution,
    _plot_split_comparison,
    _plot_token_distribution,
    analyze_and_plot_datasets,
)


FIM_MIDDLE_TOKEN_ID = 10
EOS_TOKEN_ID = 2

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"


# --- Fixtures ---

@pytest.fixture
def config(tmp_path) -> Config:
    """Load a Config from the test YAML, redirecting outputs to tmp_path."""
    test_config = Config.load_from_yaml(test_config_path)
    test_config.workspace_path = tmp_path
    test_config.raw_data_path = None  # set to none, recalculate it in _setup_paths()
    test_config._setup_paths()  # regenerates paths relative to the new workspace_path
    return test_config


@pytest.fixture
def dataset_examples():
    return [
        {"input_ids": [1, FIM_MIDDLE_TOKEN_ID, 11, 12, EOS_TOKEN_ID]},
        {"input_ids": [1, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID]},
        {"input_ids": [1, FIM_MIDDLE_TOKEN_ID, 11]},
        {"input_ids": []},
    ]


@pytest.fixture
def dataset_file(config, dataset_examples):
    _ensure_output_paths_exist(config)
    path = config.train_dataset_path
    
    with path.open("w", encoding="utf-8") as f:
        for example in dataset_examples:
            f.write(json.dumps(example) + "\n")
        f.write("\n")
    return path


@pytest.fixture
def stats_arrays():
    return {
        "train": np.array([3, 5, 7]),
        "eval": np.array([4, 6]),
        "test": np.array([2, 8, 9, 10]),
    }


# --- _extract_middle_length ---

def test_extract_middle_length_returns_length():
    input_ids = [0, FIM_MIDDLE_TOKEN_ID, 7, 8, 4, EOS_TOKEN_ID]
    assert _extract_middle_length(input_ids, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID) == 3


def test_extract_middle_length_returns_none_when_missing_tokens():
    assert _extract_middle_length([1, 2, 3], FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID) is None
    assert _extract_middle_length([1, FIM_MIDDLE_TOKEN_ID, 3, 4], FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID) is None


# --- _load_dataset_stats ---

def test_load_dataset_stats_computes_token_and_middle_lengths(dataset_file):
    result = _load_dataset_stats(dataset_file, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID)

    assert np.array_equal(result["token_lengths_np"], np.array([5, 3, 3, 0]))
    assert np.array_equal(result["middle_lengths_np"], np.array([2, 0]))


def test_load_dataset_stats_skips_blank_lines_and_missing_middle(dataset_file):
    result = _load_dataset_stats(dataset_file, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID)

    assert len(result["token_lengths_np"]) == 4
    assert len(result["middle_lengths_np"]) == 2


# --- Plotting Functions ---

def test_plot_token_distribution_saves_file(config, stats_arrays):
    _ensure_output_paths_exist(config)
    config.max_token_sequence_length = 12
    output_path = config.preprocess_results_path / "token_length_distribution.png"

    _plot_token_distribution(
        config,
        stats_arrays["train"],
        stats_arrays["eval"],
        stats_arrays["test"],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_middle_distribution_saves_file(config, stats_arrays):
    _ensure_output_paths_exist(config)
    config.min_middle_tokens_length = 2
    config.max_middle_tokens_length = 8
    output_path = config.preprocess_results_path / "middle_token_length_distribution.png"

    _plot_middle_distribution(
        config,
        stats_arrays["train"],
        stats_arrays["eval"],
        stats_arrays["test"],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_split_comparison_saves_file(config, stats_arrays):
    _ensure_output_paths_exist(config)
    output_path = config.preprocess_results_path / "split_sizes.png"

    _plot_split_comparison(
        stats_arrays["train"],
        stats_arrays["eval"],
        stats_arrays["test"],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- analyze_and_plot_datasets ---

def test_analyze_and_plot_datasets_creates_all_plots(config, dataset_examples):
    _ensure_output_paths_exist(config)

    dataset_paths = (config.train_dataset_path, config.eval_dataset_path, config.test_dataset_path)
    for path in dataset_paths:
        with path.open("w", encoding="utf-8") as f:
            for example in dataset_examples:
                f.write(json.dumps(example) + "\n")

    config.max_token_sequence_length = 12
    config.min_middle_tokens_length = 1
    config.max_middle_tokens_length = 8

    analyze_and_plot_datasets(config, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID)

    token_dist = config.preprocess_results_path / "token_length_distribution.png"
    middle_dist = config.preprocess_results_path / "middle_token_length_distribution.png"
    split_sizes = config.preprocess_results_path / "split_sizes.png"

    assert token_dist.exists() and token_dist.stat().st_size > 0
    assert middle_dist.exists() and middle_dist.stat().st_size > 0
    assert split_sizes.exists() and split_sizes.stat().st_size > 0
