import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


def _extract_middle_length(input_ids_list: list[int], fim_middle_token_id: int, eos_token_id: int) -> int | None:
    """Extract the length of the FIM middle section from input_ids."""
    try:
        middle_start_idx = input_ids_list.index(fim_middle_token_id) + 1
        middle_end_idx = input_ids_list.index(eos_token_id, middle_start_idx)
        return middle_end_idx - middle_start_idx
    except ValueError:
        logger.warning(f"Skipping example in _extract_middle_length")
        return None


def _load_dataset_stats(dataset_path: Path, fim_middle_token_id: int, eos_token_id: int) -> dict:
    token_lengths_list = []
    middle_lengths_list = []
    
    with open(dataset_path, "r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            line = line.strip()
            if not line:
                continue
            
            example_data = json.loads(line)
            input_ids_list = example_data.get("input_ids", [])
            token_lengths_list.append(len(input_ids_list))
            
            middle_length = _extract_middle_length(input_ids_list, fim_middle_token_id, eos_token_id)
            if middle_length is not None:
                middle_lengths_list.append(middle_length)
                
    return {
        "token_lengths_np": np.array(token_lengths_list),
        "middle_lengths_np": np.array(middle_lengths_list)
    }


def _plot_token_distribution(
    config: Config,
    train_np: np.ndarray,
    eval_np: np.ndarray,
    test_np: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle("Token Length Distribution Comparison", fontsize=16)
    
    data_splits = [
        (train_np, "Train", "steelblue"),
        (eval_np, "Eval", "darkorange"),
        (test_np, "Test", "green")
    ]
    
    bins_np = np.linspace(0, config.max_token_sequence_length, 60)
    
    for ax, (data, label, color) in zip(axes, data_splits):
        ax.hist(data, bins=bins_np, alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
        ax.axvline(config.max_token_sequence_length, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{label} (n={len(data)})")
        ax.set_xlabel("Token Length")
        ax.set_ylabel("Count")
        # Add 15% headroom for titles/labels
        if len(data) > 0:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_middle_distribution(
    config: Config,
    train_mid_np: np.ndarray,
    eval_mid_np: np.ndarray,
    test_mid_np: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    fig.suptitle("FIM Middle Token Length Comparison", fontsize=16)
    
    data_splits = [
        (train_mid_np, "Train", "steelblue"),
        (eval_mid_np, "Eval", "darkorange"),
        (test_mid_np, "Test", "green")
    ]
    
    max_middle = config.max_middle_tokens_length
    bins_np = np.linspace(0, max_middle + 20, 60)
    
    for ax, (data, label, color) in zip(axes, data_splits):
        ax.hist(data, bins=bins_np, alpha=0.8, color=color, edgecolor='black', linewidth=0.5)
        ax.axvline(config.min_middle_tokens_length, color="orange", linestyle="--", label="Min")
        ax.axvline(config.max_middle_tokens_length, color="red", linestyle="--", label="Max")
        ax.set_title(f"{label} (n={len(data)})")
        ax.set_xlabel("FIM Length")
        ax.set_ylabel("Count")
        if len(data) > 0:
            ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_split_comparison(
    train_np: np.ndarray,
    eval_np: np.ndarray,
    test_np: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    split_names = ["Train", "Eval", "Test"]
    split_counts = [len(train_np), len(eval_np), len(test_np)]
    
    bars = ax.bar(split_names, split_counts, color=["steelblue", "darkorange", "green"], edgecolor='black', alpha=0.9)
    
    # Increase Y-limit significantly (25% extra) to prevent text touching top
    max_count = max(split_counts) if split_counts else 1
    ax.set_ylim(0, max_count * 1.25) 
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + (max_count * 0.02),
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontweight='bold'
        )
        
    ax.set_ylabel("Number of Examples")
    ax.set_title("Dataset Split Sizes", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def analyze_and_plot_datasets(config: Config, fim_middle_token_id: int, eos_token_id: int) -> None:
    results_dir = config.preprocess_results_path

    train_stats = _load_dataset_stats(config.train_dataset_path, fim_middle_token_id, eos_token_id)
    eval_stats = _load_dataset_stats(config.eval_dataset_path, fim_middle_token_id, eos_token_id)
    test_stats = _load_dataset_stats(config.test_dataset_path, fim_middle_token_id, eos_token_id)

    _plot_token_distribution(
        config,
        train_stats["token_lengths_np"],
        eval_stats["token_lengths_np"],
        test_stats["token_lengths_np"],
        results_dir / "token_length_distribution.png",
    )
    
    _plot_middle_distribution(
        config,
        train_stats["middle_lengths_np"],
        eval_stats["middle_lengths_np"],
        test_stats["middle_lengths_np"],
        results_dir / "middle_token_length_distribution.png",
    )
    
    _plot_split_comparison(
        train_stats["token_lengths_np"],
        eval_stats["token_lengths_np"],
        test_stats["token_lengths_np"],
        results_dir / "split_sizes.png",
    )
    