import logging
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from .config import Config


logger = logging.getLogger(__name__)


def _analyze_metric_performance(config: Config, score_name: str, plot_file: Path, higher_is_better: bool) -> dict:
    base_scores = []
    lora_scores = []
    
    with open(config.evaluation_results_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if score_name == config.codebleu_score_name and not data.get("codebleu_valid", True):
                continue
            base_scores.append(data[f'base_{score_name}'])
            lora_scores.append(data[f'lora_{score_name}'])
    
    if not base_scores:
        logger.error(f"No {score_name} results found.")
        return {}
    
    base_arr = np.array(base_scores)
    lora_arr = np.array(lora_scores)
    n_examples = len(base_arr)
    
    avg_base = np.mean(base_arr)
    avg_lora = np.mean(lora_arr)
    improvement = (avg_lora - avg_base) if higher_is_better else (avg_base - avg_lora)
    
    logger.info(f"\n=== {score_name.upper()} SUMMARY ===")
    logger.info(f"Examples: {n_examples}")
    logger.info(f"Base avg: {avg_base:.3f}")
    logger.info(f"LoRA avg: {avg_lora:.3f}")
    logger.info(f"Improvement (signed): {improvement:+.3f}")
    
    is_binary = score_name in [config.exact_match_score_name, config.line_match_score_name]
    
    if is_binary:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(['Base', 'LoRA'], [avg_base, avg_lora], color=['#1f77b4', '#ff7f0e'])
        ax1.set_title(f'{score_name.title()} Success Rate')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)
        
        wins = np.sum((lora_arr == 1) & (base_arr == 0))
        losses = np.sum((lora_arr == 0) & (base_arr == 1))
        ties = n_examples - wins - losses
        ax2.bar(['LoRA Wins', 'Ties', 'LoRA Losses'], [wins, ties, losses], color=['green', 'gray', 'red'])
        ax2.set_title('Example-Level Transitions')
        ax2.set_ylabel('Count')
    else:
        y_axis_limit = 25
        # Calculate dynamic bounds
        all_data = np.concatenate([base_arr, lora_arr])
        max_val = np.max(all_data)
        
        # Determine if we need to enforce the limit
        use_limit = max_val > y_axis_limit
        upper_bound = y_axis_limit if use_limit else max_val * 1.1
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Bar Plot (Unchanged)
        axs[0, 0].bar(['Base', 'LoRA'], [avg_base, avg_lora], color=['steelblue', 'darkorange'])
        axs[0, 0].set_title('Average Scores')
        
        # 2. Boxplot
        axs[0, 1].boxplot([base_arr, lora_arr], labels=['Base', 'LoRA'], patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=2))
        axs[0, 1].set_title('Score Distribution')
        if use_limit:
            axs[0, 1].set_ylim(0, upper_bound)
            axs[0, 1].text(0.95, 0.95, f'Note: Values > {y_axis_limit} hidden', 
                          transform=axs[0, 1].transAxes, ha='right', va='top', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8))
        
        # 3. Scatter Plot
        axs[1, 0].scatter(base_arr, lora_arr, alpha=0.5, edgecolors='none', color='steelblue')
        axs[1, 0].plot([0, upper_bound], [0, upper_bound], 'k--', alpha=0.75, label='y=x (Tie)')
        axs[1, 0].set_title('Base vs LoRA (Per Example)')
        axs[1, 0].set_xlim(0, upper_bound)
        axs[1, 0].set_ylim(0, upper_bound)
        if use_limit:
            axs[1, 0].text(0.95, 0.05, f'Note: Values > {y_axis_limit} hidden', 
                          transform=axs[1, 0].transAxes, ha='right', va='bottom', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8))
        axs[1, 0].legend(loc='upper left')
        
        # 4. Histogram
        differences = (lora_arr - base_arr) if higher_is_better else (base_arr - lora_arr)
        hist_range = (-upper_bound, upper_bound) if use_limit else None
        axs[1, 1].hist(differences, bins=30, range=hist_range, color='purple', alpha=0.7, edgecolor='black')
        axs[1, 1].axvline(0, color='black', linestyle='dashed', linewidth=1)
        axs[1, 1].axvline(np.mean(differences), color='red', linestyle='solid', linewidth=2, label='Mean Diff')
        axs[1, 1].set_title('Improvement Distribution')
        if use_limit:
            axs[1, 1].set_xlim(hist_range)
            axs[1, 1].text(0.95, 0.95, f'Note: |Diff| > {y_axis_limit} hidden', 
                          transform=axs[1, 1].transAxes, ha='right', va='top', fontsize=10, 
                          bbox=dict(facecolor='white', alpha=0.8))
        axs[1, 1].legend(loc='upper left')

    plt.suptitle(f'{score_name.upper()} Evaluation (N={n_examples})', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Updated plot saved: {plot_file}")
    
    return {
        "metric": score_name,
        "examples": n_examples,
        "base_avg": float(avg_base),
        "lora_avg": float(avg_lora),
        "improvement": float(improvement)
    }


def _save_evaluation_report(config: Config, checkpoint_name: str, metric_results: list[dict]) -> None:
    """Writes all processed metrics to a single summary JSON file."""
    report_path = config.evaluation_report_path
    report_content = {
        "checkpoint": checkpoint_name,
        "evaluation_date": datetime.now().isoformat(timespec="seconds"),
        "results": metric_results
    }
    with open(report_path, "w") as f:
        json.dump(report_content, f, indent=4)
    logger.info(f"Summary report saved to: {report_path}")


def _plot_all_metric_averages(evaluation_report_path: Path, output_file: Path) -> None:
    if not evaluation_report_path.exists():
        logger.error(f"Evaluation report file not found: {evaluation_report_path}")
        return

    with open(evaluation_report_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        logger.error("No results found in evaluation report.")
        return

    n_metrics = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        metric_name = result["metric"]
        base_avg = result["base_avg"]
        lora_avg = result["lora_avg"]
        n_examples = result["examples"]
        
        ax = axes[idx]
        bars = ax.bar(['Base', 'LoRA'], [base_avg, lora_avg], 
                      color=['steelblue', 'darkorange'], alpha=0.8)
        
        max_val = max(base_avg, lora_avg) * 1.1
        ax.set_ylim(0, max_val)
        
        ax.set_ylabel('Score')
        ax.set_title(f'{metric_name.title()}\nN={n_examples}')
        
        for bar, val in zip(bars, [base_avg, lora_avg]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_val*0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11)
    
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Base vs LoRA Average Scores', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"All-metrics average subplor saved: {output_file}")