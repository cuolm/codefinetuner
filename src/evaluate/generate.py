import argparse
import gc
import json
import logging
import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint

from .config import Config


logger = logging.getLogger("stc.evaluate.generate")


def _load_model(config: Config, user_args: argparse.Namespace) ->AutoModelForCausalLM:
    if user_args.checkpoint == "last":
        checkpoint_path = get_last_checkpoint(config.trainer_output_dir_path)
    else:
        checkpoint_path = config.trainer_output_dir_path / user_args.checkpoint
    
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

    logger.info(f"Loaded base model and LoRA adapter: {checkpoint_path}")
    return model


def _load_tokenizer(config: Config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "right"
    return tokenizer


def _get_fim_perplexity(config: Config, model: AutoModelForCausalLM, 
                        perplexity_input_token_ids: list, perplexity_label_token_ids) -> float:
    """
    FIM perplexity: Measures model confidence in the ground truth reference middle code.
    (How surprised is the model by the ground truth reference middle code).
    Lower perplexity indicates higher confidence. perplexity = exp(loss).
    """
    try:
        input_tensor = torch.tensor([perplexity_input_token_ids], device=config.device)
        label_tensor = torch.tensor([perplexity_label_token_ids], device=config.device)
        with torch.inference_mode():
            outputs = model(input_ids=input_tensor, 
                            labels=label_tensor)
            loss = outputs.loss
            
        return math.exp(loss.item())
        
    except Exception as e:
        logger.exception(f"ERROR in perplexity calculation: {e}")
        return float('inf')


def _generate(config: Config, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_token_ids: list) -> list:
    model.eval()
    prompt_token_ids_tensor = torch.tensor([prompt_token_ids], device=config.device)
    with torch.inference_mode():
        generated_token_ids_tensor = model.generate(
            input_ids=prompt_token_ids_tensor,
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=config.generation_do_sample,
            temperature=config.generation_temperature,
            top_p=config.generation_top_p,
            pad_token_id=tokenizer.pad_token_id
        )

        # slice the output, take everything after the input_length, model.generate() functin returns the whole example, not only the generated text
        generated_middle_token_ids_tensor = generated_token_ids_tensor[0][prompt_token_ids_tensor.shape[1] :]
        generated_middle_token_ids = generated_middle_token_ids_tensor.tolist()

    return generated_middle_token_ids


def _clear_hardware_cache(config: Config) -> None:
    gc.collect()
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif config.device == "mps":
        torch.mps.empty_cache()


def generate_and_save(config: Config, user_args: argparse.Namespace):
    model = _load_model(config, user_args)
    tokenizer = _load_tokenizer(config)
    fim_prefix_token_id = tokenizer.convert_tokens_to_ids(config.fim_prefix_token)
    fim_suffix_token_id = tokenizer.convert_tokens_to_ids(config.fim_suffix_token)
    fim_middle_token_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)
    
    try:
        line_counter = 0
        with config.benchmark_dataset_path.open("r") as benchmark_dataset_file, \
            config.benchmark_evaluation_results_path.open("w") as evaluation_results_file:
            
            for line in benchmark_dataset_file:
                benchmark_example = json.loads(line)
                prompt_token_ids = ([fim_prefix_token_id] + benchmark_example["prefix_token_ids"] +
                                   [fim_suffix_token_id] + benchmark_example["suffix_token_ids"] +
                                   [fim_middle_token_id])
                
                perplexity_input_token_ids = benchmark_example["example_token_ids"]
                perplexity_label_token_ids = perplexity_input_token_ids.copy()
                perplexity_label_token_ids[:len(prompt_token_ids)] = [-100] * len(prompt_token_ids)  # mask labels all except ground truth reference middle tokens

                lora_generated_middle_token_ids = _generate(config, model, tokenizer, prompt_token_ids)
                lora_perplexity = _get_fim_perplexity(config, model, perplexity_input_token_ids, perplexity_label_token_ids)

                with model.disable_adapter():
                    base_generated_middle_token_ids = _generate(config, model, tokenizer, prompt_token_ids)
                    base_perplexity = _get_fim_perplexity(config, model, perplexity_input_token_ids, perplexity_label_token_ids)

                reference_middle = benchmark_example["middle"]
                lora_generated_middle = tokenizer.decode(lora_generated_middle_token_ids, skip_special_tokens=True)
                base_generated_middle = tokenizer.decode(base_generated_middle_token_ids, skip_special_tokens=True)
                result = {
                    "example_id": line_counter,
                    "reference_middle": reference_middle,
                    "base_generated_middle": base_generated_middle,
                    "lora_generated_middle": lora_generated_middle,
                    "base_perplexity": base_perplexity,
                    "lora_perplexity": lora_perplexity
                }
                evaluation_results_file.write(json.dumps(result) + "\n")

                line_counter += 1
                if line_counter % 10 == 0:
                    _clear_hardware_cache(config)
                    logger.info(f"Processed Example {line_counter}")
        
        logger.info(f"Successfully generated and saved {line_counter} number of examples to {config.benchmark_evaluation_results_path}.")

    except Exception:
        logger.exception("Generation failed.")
        raise
    