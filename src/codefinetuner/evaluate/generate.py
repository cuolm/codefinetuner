import gc
import json
import logging
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import Config


logger = logging.getLogger(__name__)


def _load_lora_model(config: Config, checkpoint_path: Path):
    if config.use_unsloth:
        return _load_unsloth_lora_model(config, checkpoint_path)
    else:
        return _load_hf_lora_model(config, checkpoint_path)


def _load_hf_lora_model(config: Config, checkpoint_path: Path):
    # load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config.model_name,
        dtype=config.model_dtype,
        low_cpu_mem_usage=True
    )

    # attach LoRA adapter to base model
    lora_model = PeftModel.from_pretrained(
        model=base_model,
        model_id=checkpoint_path
    ).to(config.device)
    lora_model.eval()
    logger.info(f"Loaded HF LoRA model from checkpoint {checkpoint_path} to device {config.device}")
    return lora_model


def _load_unsloth_lora_model(config: Config, checkpoint_path: Path):
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError("Unsloth is not installed.")

    # load base model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_token_sequence_length,
        dtype=config.model_dtype,
        load_in_4bit=True,
    )

    # attach LoRA adapter to base model 
    from peft import PeftModel
    lora_model = PeftModel.from_pretrained(
        model=model,
        model_id=checkpoint_path
    )
    lora_model.eval()
    logger.info(f"Loaded Unsloth LoRA model from {checkpoint_path}")
    return lora_model


def _load_tokenizer(config: Config) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = config.fim_pad_token
    tokenizer.padding_side = "left"
    return tokenizer


def _get_fim_perplexities(config: Config, model: AutoModelForCausalLM,
                            perplexity_input_token_ids_batch: list[list[int]], perplexity_label_token_ids_batch: list[list[int]]) -> list[float]:
    """
    FIM perplexity: Measures model confidence in the ground truth reference middle code.
    (How surprised is the model by the ground truth reference middle code).
    Lower perplexity indicates higher confidence. perplexity = exp(loss).
    """
    perplexities = []
    for input_ids, label_ids in zip(perplexity_input_token_ids_batch, perplexity_label_token_ids_batch):
        input_tensor = torch.tensor([input_ids], device=config.device)
        label_tensor = torch.tensor([label_ids], device=config.device)
        
        with torch.inference_mode():
            outputs = model(input_ids=input_tensor, labels=label_tensor)
            loss = outputs.loss
            perplexities.append(math.exp(loss.item()))
    return perplexities


def _generate(config: Config, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_token_ids_list: list[list[int]]) -> list[list[int]]:
    model.eval()
    
    # 1. Manually pad the sequences to the longest one in the batch
    # tokenizer.pad expects a dictionary format for return_tensors
    encoded_inputs = tokenizer.pad(
        {"input_ids": prompt_token_ids_list},
        padding=True,
        return_tensors="pt"
    ).to(config.device)

    input_ids = encoded_inputs["input_ids"]
    attention_mask = encoded_inputs["attention_mask"]

    with torch.inference_mode():
        generated_token_ids_tensor = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, # Critical for batching
            max_new_tokens=config.generation_max_new_tokens,
            do_sample=config.generation_do_sample,
            temperature=config.generation_temperature,
            top_p=config.generation_top_p,
            pad_token_id=tokenizer.pad_token_id
        )

        # 2. Slice the output for the entire batch
        # With left padding, the prompt length is the same for all in the padded tensor
        prompt_length = input_ids.shape[1]
        generated_middle_batch = generated_token_ids_tensor[:, prompt_length:]

    return generated_middle_batch.tolist()


def _clear_hardware_cache(config: Config) -> None:
    gc.collect()
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif config.device == "mps":
        torch.mps.empty_cache()


def generate_and_save(config: Config, checkpoint_path: Path):
    lora_model = _load_lora_model(config, checkpoint_path)
    tokenizer = _load_tokenizer(config)
    fim_prefix_token_id = tokenizer.convert_tokens_to_ids(config.fim_prefix_token)
    fim_suffix_token_id = tokenizer.convert_tokens_to_ids(config.fim_suffix_token)
    fim_middle_token_id = tokenizer.convert_tokens_to_ids(config.fim_middle_token)

    batch_examples = []
    prompt_token_ids_batch  = []
    perplexity_input_token_ids_batch = []
    perplexity_label_token_ids_batch = []

    def _process_batch():
        lora_generated_middle_token_ids_batch = _generate(config, lora_model, tokenizer, prompt_token_ids_batch)
        lora_perplexities_batch = _get_fim_perplexities(config, lora_model, perplexity_input_token_ids_batch, perplexity_label_token_ids_batch)

        with lora_model.disable_adapter():
            base_generated_middle_token_ids_batch = _generate(config, lora_model, tokenizer, prompt_token_ids_batch)
            base_perplexities_batch = _get_fim_perplexities(config, lora_model, perplexity_input_token_ids_batch, perplexity_label_token_ids_batch)

        zipped_data = zip(
            batch_examples, 
            base_generated_middle_token_ids_batch, 
            lora_generated_middle_token_ids_batch, 
            base_perplexities_batch, 
            lora_perplexities_batch
        )

        for example, base_generated_middle_token_ids, lora_generated_middle_token_ids, base_ppl, lora_ppl in zipped_data:
            base_generated_middle = tokenizer.decode(base_generated_middle_token_ids, skip_special_tokens=True)
            lora_generated_middle = tokenizer.decode(lora_generated_middle_token_ids, skip_special_tokens=True)
            base_generated_middle = base_generated_middle.replace(config.fim_pad_token, "").strip()
            lora_generated_middle = lora_generated_middle.replace(config.fim_pad_token, "").strip() 
            
            result = {
                "reference_middle": example["middle"], 
                "base_generated_middle": base_generated_middle,
                "lora_generated_middle": lora_generated_middle,
                "base_perplexity": base_ppl,
                "lora_perplexity": lora_ppl
            }
            evaluation_results_file.write(json.dumps(result) + "\n")

        batch_examples.clear()
        prompt_token_ids_batch.clear() 
        perplexity_input_token_ids_batch.clear()
        perplexity_label_token_ids_batch.clear()
        _clear_hardware_cache(config)

    line_counter = 0 
    try:
        with config.benchmark_dataset_path.open("r") as benchmark_dataset_file, \
            config.benchmark_evaluation_results_path.open("w") as evaluation_results_file:

            for line in benchmark_dataset_file:
                benchmark_example = json.loads(line)
                batch_examples.append(benchmark_example)
                prompt_token_ids = ([fim_prefix_token_id] + benchmark_example["prefix_token_ids"] +
                                    [fim_suffix_token_id] + benchmark_example["suffix_token_ids"] +
                                    [fim_middle_token_id])
                prompt_token_ids_batch.append(prompt_token_ids)
                
                perplexity_input_token_ids = benchmark_example["example_token_ids"]
                perplexity_label_token_ids = perplexity_input_token_ids.copy()
                perplexity_label_token_ids[:len(prompt_token_ids)] = [config.label_pad_token_id] * len(prompt_token_ids)  # mask labels all except ground truth reference middle tokens
                perplexity_input_token_ids_batch.append(perplexity_input_token_ids)
                perplexity_label_token_ids_batch.append(perplexity_label_token_ids)

                line_counter += 1

                if len(prompt_token_ids_batch) == config.generation_batch_size:
                    _process_batch()
                    logger.info(f"Processed {line_counter} benchmark examples")

            if batch_examples:
                _process_batch()
                logger.info(f"Processed final {len(batch_examples)} examples. Total: {line_counter}")
        
    except Exception as e:
        raise RuntimeError(f"Generation failed at example {line_counter}: {e}") from e
    
    logger.info(f"Successfully generated and saved {line_counter} number of examples to {config.benchmark_evaluation_results_path}.")
