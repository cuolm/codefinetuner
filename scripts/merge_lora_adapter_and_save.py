import argparse
from pathlib import Path
from peft import AutoPeftModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter using AutoPeftModel")
    parser.add_argument("lora_adapter_path", type=Path, help="Path to LoRA adapter directory")
    parser.add_argument("--output_dir", type=Path, default="./lora_model", help="Output directory")
    
    args = parser.parse_args()
    
    model = AutoPeftModelForCausalLM.from_pretrained(args.lora_adapter_path)
    model = model.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    
    print(f"Merged {args.lora_adapter_path} → {args.output_dir}")


if __name__ == "__main__":
    main()
