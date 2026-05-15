import json
import pathlib

import numpy as np
import pytest

from codefinetuner.preprocess.analyze import (
    _extract_middle_length,
    _load_dataset_stats,
    _plot_middle_distribution,
    _plot_split_comparison,
    _plot_token_distribution,
    analyze_and_plot_datasets,
)
from codefinetuner.preprocess.config import Config


FIM_MIDDLE_TOKEN_ID = 10
EOS_TOKEN_ID = 2

tests_path = pathlib.Path(__file__).parent.parent
test_config_path = tests_path / "config" / "codefinetuner_config.yaml"


@pytest.fixture
def config() -> Config:
    return Config.load_from_yaml(test_config_path)


@pytest.fixture
def dataset_examples():
    return [
        {"input_ids": [1, FIM_MIDDLE_TOKEN_ID, 11, 12, EOS_TOKEN_ID]},
        {"input_ids": [1, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID]},
        {"input_ids": [1, FIM_MIDDLE_TOKEN_ID, 11]},
        {"input_ids": []},
    ]


@pytest.fixture
def dataset_file(tmp_path, dataset_examples):
    path = tmp_path / "dataset.jsonl"
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


def test_extract_middle_length_returns_length():
    input_ids = [0, FIM_MIDDLE_TOKEN_ID, 7, 8, 4, EOS_TOKEN_ID]
    assert _extract_middle_length(input_ids, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID) == 3


def test_extract_middle_length_returns_none_when_missing_tokens():
    assert _extract_middle_length([1, 2, 3], FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID) is None
    assert _extract_middle_length([1, FIM_MIDDLE_TOKEN_ID, 3, 4], FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID) is None


def test_load_dataset_stats_computes_token_and_middle_lengths(dataset_file):
    result = _load_dataset_stats(dataset_file, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID)

    assert np.array_equal(result["token_lengths_np"], np.array([5, 3, 3, 0]))
    assert np.array_equal(result["middle_lengths_np"], np.array([2, 0]))


def test_load_dataset_stats_skips_blank_lines_and_missing_middle(dataset_file):
    result = _load_dataset_stats(dataset_file, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID)

    assert len(result["token_lengths_np"]) == 4
    assert len(result["middle_lengths_np"]) == 2


def test_plot_token_distribution_saves_file(config, tmp_path, stats_arrays):
    config.max_token_sequence_length = 12
    output_path = tmp_path / "token_length_distribution.png"

    _plot_token_distribution(
        config,
        stats_arrays["train"],
        stats_arrays["eval"],
        stats_arrays["test"],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_middle_distribution_saves_file(config, tmp_path, stats_arrays):
    config.min_middle_tokens_length = 2
    config.max_middle_tokens_length = 8
    output_path = tmp_path / "middle_token_length_distribution.png"

    _plot_middle_distribution(
        config,
        stats_arrays["train"],
        stats_arrays["eval"],
        stats_arrays["test"],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_split_comparison_saves_file(tmp_path, stats_arrays):
    output_path = tmp_path / "split_sizes.png"

    _plot_split_comparison(
        stats_arrays["train"],
        stats_arrays["eval"],
        stats_arrays["test"],
        output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_analyze_and_plot_datasets_creates_all_plots(config, tmp_path, dataset_examples):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    test_path = tmp_path / "test.jsonl"

    for path in (train_path, eval_path, test_path):
        with path.open("w", encoding="utf-8") as f:
            for example in dataset_examples:
                f.write(json.dumps(example) + "\n")

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    config.train_dataset_path = train_path
    config.eval_dataset_path = eval_path
    config.test_dataset_path = test_path
    config.preprocess_results_path = results_dir
    config.max_token_sequence_length = 12
    config.min_middle_tokens_length = 1
    config.max_middle_tokens_length = 8

    analyze_and_plot_datasets(config, FIM_MIDDLE_TOKEN_ID, EOS_TOKEN_ID)

    assert (results_dir / "token_length_distribution.png").exists()
    assert (results_dir / "middle_token_length_distribution.png").exists()
    assert (results_dir / "split_sizes.png").exists()
    assert (results_dir / "token_length_distribution.png").stat().st_size > 0
    assert (results_dir / "middle_token_length_distribution.png").stat().st_size > 0
    assert (results_dir / "split_sizes.png").stat().st_size > 0