from transformers.trainer_utils import get_last_checkpoint
import logging
import argparse
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


from .config import Config

from .generation import _generate_code, _clear_hardware_cache
from .metrics import _get_codebleu, _get_sentencebleu, _get_exact_match, _get_line_match, _get_fim_perplexity

from .utils import _setup_logger, _parse_args, _ensure_benchmark_dataset, _ensure_directories_exist
from .analysis import _analyze_metric_performance, _save_evaluation_report, _plot_all_metric_averages

logger = logging.getLogger(__name__)


def _evaluate_models_to_file(config: Config, user_args: argparse.Namespace, tokenizer: AutoTokenizer) -> None:
    if user_args.checkpoint == "last":
        checkpoint_path = get_last_checkpoint(config.trainer_output_dir_path)
    else:
        checkpoint_path = config.trainer_output_dir_path / user_args.checkpoint

    logger.info(f"--- Loading Base Model and LoRA Adapter: {checkpoint_path} ---")
    
    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        dtype=torch.float16,
        device_map="auto" if config.device == "cuda" else None,
        low_cpu_mem_usage=True
    )
    if config.device != "cuda":
        base_model.to(config.device)

    # load LoRA adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    logger.info("--- Starting Generation and Evaluation ---")
    
    # process each example 
    with config.benchmark_dataset_path.open("r") as benchmark_dataset_file, \
         config.evaluation_results_path.open("w") as evaluation_results_file:
        
        for i, line in enumerate(benchmark_dataset_file):
            example = json.loads(line)
            prompt = (
                f"{config.fim_prefix_token}{example['prefix']}"
                f"{config.fim_suffix_token}{example['suffix']}"
                f"{config.fim_middle_token}"
            )
            reference_middle = example["reference_middle"]

            # generation with LoRA augmented model
            lora_generated_middle = _generate_code(config, model, tokenizer, prompt)
            lora_perplexity = _get_fim_perplexity(config, model, tokenizer, example["prefix"], example["suffix"], reference_middle)

            # generation with base model by disabling lora adapter 
            with model.disable_adapter():
                base_generated_middle = _generate_code(config, model, tokenizer, prompt)
                base_perplexity = _get_fim_perplexity(config, model, tokenizer, example["prefix"], example["suffix"], reference_middle)

            # metrics calculation 
            base_codebleu, codebleu_valid = _get_codebleu(config, reference_middle, base_generated_middle)
            lora_codebleu, _ = _get_codebleu(config, reference_middle, lora_generated_middle)
            
            base_sentencebleu = _get_sentencebleu(config, reference_middle, base_generated_middle)
            lora_sentencebleu = _get_sentencebleu(config, reference_middle, lora_generated_middle)

            base_exact_match = _get_exact_match(config, reference_middle, base_generated_middle)
            lora_exact_match = _get_exact_match(config, reference_middle, lora_generated_middle)

            base_line_match = _get_line_match(config, reference_middle, base_generated_middle)
            lora_line_match = _get_line_match(config, reference_middle, lora_generated_middle)

            result = {
                "example_id": i,
                "reference_middle": reference_middle,
                "base_generated_middle": base_generated_middle,
                "lora_generated_middle": lora_generated_middle,
                "base_codebleu": base_codebleu,
                "lora_codebleu": lora_codebleu,
                "codebleu_valid": codebleu_valid,
                "base_sentencebleu": base_sentencebleu,
                "lora_sentencebleu": lora_sentencebleu,
                "base_exact_match": base_exact_match,
                "lora_exact_match": lora_exact_match,
                "base_line_match": base_line_match,
                "lora_line_match": lora_line_match,
                "base_perplexity": base_perplexity,
                "lora_perplexity": lora_perplexity
            }
            evaluation_results_file.write(json.dumps(result) + "\n")

            if i % 10 == 0:
                _clear_hardware_cache(config)
                logger.info(f"Processed Example {i}")

    del model
    del base_model
    _clear_hardware_cache(config)



def main():
    config = Config()
    _setup_logger("INFO")
    user_args = _parse_args()

    _ensure_benchmark_dataset(config, user_args)
    
    if not user_args.plot_only:  
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = config.fim_pad_token
        tokenizer.padding_side = "right"

        _ensure_directories_exist(config)
        _evaluate_models_to_file(config, user_args, tokenizer)

    metrics_configurations = [
        (config.sentencebleu_score_name, config.sentencebleu_plot_file, True),
        (config.codebleu_score_name, config.codebleu_plot_file, True),
        (config.exact_match_score_name, config.exact_match_plot_file, True),
        (config.line_match_score_name, config.line_match_plot_file, True),
        (config.perplexity_name, config.perplexity_plot_file, False),
    ]

    all_metric_stats = []
    for score_name, plot_path, higher_is_better in metrics_configurations:
        stats = _analyze_metric_performance(config, score_name, plot_path, higher_is_better)
        if stats:
            all_metric_stats.append(stats)
    
    _save_evaluation_report(config, user_args.checkpoint, all_metric_stats)
    _plot_all_metric_averages(config.evaluation_report_path, config.all_metrics_average) 


if __name__ == "__main__":
    main()