import json

def analyze_token_lengths(file_path, threshold):
    max_len = 0
    total_count = 0
    under_threshold_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                total_count += 1
                data = json.loads(line)
                current_len = len(data.get("input_ids", []))
                
                if current_len > max_len:
                    max_len = current_len
                
                if current_len <= threshold:
                    under_threshold_count += 1
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Total examples: {total_count}")
    print(f"Maximum token length: {max_len}")
    print(f"Examples under/at threshold ({threshold}): {under_threshold_count} out of {total_count}")

if __name__ == "__main__":
    # CONFIGURATION
    file_path = "outputs/preprocess/results/datasets/train_dataset.jsonl"
    threshold = 512
    
    analyze_token_lengths(file_path, threshold)