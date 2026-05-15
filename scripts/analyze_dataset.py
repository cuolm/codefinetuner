import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_token_lengths(file_path: Path, threshold: int):
    lengths = []
    under_threshold_count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                current_len = len(data.get("input_ids", []))
                lengths.append(current_len)
                if current_len <= threshold:
                    under_threshold_count += 1
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    total_count = len(lengths)
    max_len = max(lengths) if lengths else 0

    print(f"Total examples: {total_count}")
    print(f"Maximum token length: {max_len}")
    print(f"Examples under/at threshold ({threshold}): {under_threshold_count} out of {total_count}")

    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, edgecolor="black")
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Token Length")
    plt.ylabel("Count")
    plt.title("Token Length Distribution")
    plt.legend()
    plt.tight_layout()
    if args.save_plot:
        plt.savefig("token_length_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, required=True)
    parser.add_argument("--threshold", type=int, default=512)
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()
    analyze_token_lengths(args.dataset_path, args.threshold)