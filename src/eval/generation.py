import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import Config

def _generate_code(config: Config, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> str:
    model.eval()
    input_tokens_dict = tokenizer(prompt, return_tensors="pt").to(config.device)

    bad_words_ids = [
        tokenizer.encode(config.fim_prefix_token, add_special_tokens=False),
        tokenizer.encode(config.fim_middle_token, add_special_tokens=False),
        tokenizer.encode(config.fim_suffix_token, add_special_tokens=False)
    ]

    with torch.inference_mode(): 
        outputs = model.generate(
                input_ids=input_tokens_dict["input_ids"],
                attention_mask=input_tokens_dict["attention_mask"], 
                max_new_tokens=config.gen_max_new_tokens,
                do_sample=config.gen_do_sample,
                temperature=config.gen_temperature,
                top_p=config.gen_top_p,
                bad_words_ids=bad_words_ids,
                pad_token_id=tokenizer.pad_token_id
            )
    

    # Slice the output: take everything after the input_length. generate functin returns the whole example, not only the generated text
    input_length = input_tokens_dict["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    generated_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_code 


def _clear_hardware_cache(config: Config) -> None:
    gc.collect()
    if config.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif config.device == "mps":
        torch.mps.empty_cache()