import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def clean_st_content(content: str) -> str:
    """Strips all block and inline comments, IDE pragmas, and normalizes empty lines."""
    content = re.sub(r"\(\*.*?\*\)", "", content, flags=re.DOTALL)
    content = re.sub(r"//.*", "", content)

    lines = [line.rstrip() for line in content.splitlines()]
    cleaned_lines = []
    for line in lines:
        if line:
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != "":
            cleaned_lines.append("")

    return "\n".join(cleaned_lines).strip()


def extract_blocks(content: str) -> List[str]:
    """Extracts all matching structured text blocks from the content, including modern OOP structures."""
    pattern = re.compile(
        r"\b(?:"
        r"FUNCTION_BLOCK\b.*?\bEND_FUNCTION_BLOCK|"
        r"FUNCTION\b.*?\bEND_FUNCTION|"
        r"PROGRAM\b.*?\bEND_PROGRAM|"
        r"TYPE\b.*?\bEND_TYPE|"
        r"CONFIGURATION\b.*?\bEND_CONFIGURATION|"
        r"VAR_GLOBAL\b.*?\bEND_VAR|"
        r"CLASS\b.*?\bEND_CLASS|"
        r"INTERFACE\b.*?\bEND_INTERFACE|"
        r"NAMESPACE\b.*?\bEND_NAMESPACE|"
        r"METHOD\b.*?\bEND_METHOD|"
        r"PROPERTY\b.*?\bEND_PROPERTY"
        r")",
        re.DOTALL | re.IGNORECASE,
    )
    return [match.group(0) for match in pattern.finditer(content)]


def categorize_blocks(blocks: List[str]) -> Dict[str, List[str]]:
    """Groups blocks into a dictionary based on their starting keyword."""
    categorized = defaultdict(list)
    for block in blocks:
        first_word = block.split(maxsplit=1)[0].upper()
        categorized[first_word].append(block)
    return categorized


def stratify_blocks(
    categorized: Dict[str, List[str]], train_ratio: float, eval_ratio: float
) -> Tuple[List[str], List[str], List[str]]:
    """Splits categorized blocks proportionally into train, eval, and test sets."""
    train, eval_set, test = [], [], []
    random.seed(42)

    for category, block_list in categorized.items():
        random.shuffle(block_list)
        total = len(block_list)

        train_end = int(total * train_ratio)
        eval_end = train_end + int(total * eval_ratio)

        train.extend(block_list[:train_end])
        eval_set.extend(block_list[train_end:eval_end])
        test.extend(block_list[eval_end:])

        print(
            f"Category {category:15} total: {total:<4} -> "
            f"Train: {len(block_list[:train_end]):<4} "
            f"Eval: {len(block_list[train_end:eval_end]):<4} "
            f"Test: {len(block_list[eval_end:]):<4}"
        )

    random.shuffle(train)
    random.shuffle(eval_set)
    random.shuffle(test)
    return train, eval_set, test


def write_splits(target_dir: Path, splits: Dict[str, List[str]]) -> None:
    """Saves the distributed block splits into their respective files."""
    print("-" * 50)
    for filename, block_list in splits.items():
        output_path = target_dir / filename
        output_path.write_text("\n\n".join(block_list) + "\n", encoding="utf-8")
        print(f"Wrote {len(block_list)} total stratified blocks to {output_path}")


def split_st_by_structure(
    input_file: Path,
    output_dir: Path,
    train_ratio: float,
    eval_ratio: float,
    test_ratio: float,
):
    """Coordinates file reading, processing, stratification, and export operations."""
    target_dir = output_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        print(f"Error: Input file not found at {input_file}")
        return

    raw_content = input_file.read_text(encoding="utf-8")
    cleaned_content = clean_st_content(raw_content)

    blocks = extract_blocks(cleaned_content)
    if not blocks:
        print("Error: No valid ST blocks found after cleaning.")
        return

    categorized = categorize_blocks(blocks)
    train, eval_set, test = stratify_blocks(categorized, train_ratio, eval_ratio)

    write_splits(
        target_dir, {"train.st": train, "eval.st": eval_set, "test.st": test}
    )


def main():
    parser = argparse.ArgumentParser(
        description="Clean clutter and split Structured Text (ST) files structurally using stratified sampling."
    )
    parser.add_argument("input_file", type=Path, help="Path to the input .st source file")
    parser.add_argument("output_dir", type=Path, help="Target directory for the clean splits")
    parser.add_argument("--train", type=float, default=0.80, help="Train ratio (default: 0.80)")
    parser.add_argument("--eval", type=float, default=0.10, help="Evaluation ratio (default: 0.10)")
    parser.add_argument("--test", type=float, default=0.10, help="Test ratio (default: 0.10)")

    args = parser.parse_args()

    if not abs((args.train + args.eval + args.test) - 1.0) < 1e-4:
        print(f"Error: Ratios must sum up to 1.0 (Current sum: {args.train + args.eval + args.test})")
        return

    split_st_by_structure(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train,
        eval_ratio=args.eval,
        test_ratio=args.test,
    )


if __name__ == "__main__":
    main()