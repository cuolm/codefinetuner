#!/usr/bin/env python3
import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32


def main():
    parser = argparse.ArgumentParser(description="Save base HF model and tokenizer")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    
    args = parser.parse_args()

    model_folder = args.model_name.split("/")[-1]
    
    project_root_path = Path(__file__).resolve().parent.parent
    output_path = project_root_path / "base_models" / model_folder
    output_path.mkdir(parents=True, exist_ok=True)
    
    device, model_dtype = get_device_dtype()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        dtype=model_dtype
    ).to(device)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    main()
