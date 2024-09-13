import logging
logger = logging.getLogger("main")

import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from utils_hh.modify_mistral import MistralForCausalLM_h2o, MistralFlashAttention2_heavy_hitter
from utils_hh.modify_llama import LlamaFlashAttention2_h2o, LlamaForCausalLM_h2o
import os
from tqdm import tqdm
from datasets import load_dataset, load_from_disk


TAGET_MODULE = {
    "mistral": MistralFlashAttention2_heavy_hitter,
    "llama": LlamaFlashAttention2_h2o,
}

def h2o_params_check(pipeline_config):
    assert pipeline_config['use_flash_attn'] 

def initialize_model_tokenizer(pipeline_config):
    h2o_params_check(pipeline_config)
    config = AutoConfig.from_pretrained(pipeline_config['model_name'])
    config.heavy_ratio = pipeline_config['heavy_ratio']
    config.recent_ratio = pipeline_config['recent_ratio']
    config._attn_implementation = "flash_attention_2"
    if "mistral" in pipeline_config['model_name'].lower():
        model = MistralForCausalLM_h2o.from_pretrained(
            pretrained_model_name_or_path=pipeline_config['model_name'],
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    elif "longchat" in pipeline_config['model_name'].lower():
        model = LlamaForCausalLM_h2o.from_pretrained(
            pretrained_model_name_or_path=pipeline_config['model_name'],
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    elif "llama-3" in pipeline_config['model_name'].lower():
        config.rope_theta = config.rope_theta * pipeline_config['rope_theta_factor']
        model = LlamaForCausalLM_h2o.from_pretrained(
            pretrained_model_name_or_path=pipeline_config['model_name'],
            config=config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        raise NotImplementedError
    tokenizer = AutoTokenizer.from_pretrained(pipeline_config['model_name'], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')
    return model, tokenizer

def batch_generate(batched_input, model, tokenizer, max_new_tokens, pipeline_config, **kwargs):
    model.eval()

    if isinstance(batched_input[0], str):
        model_inputs = tokenizer(batched_input, return_tensors="pt", padding=True).to("cuda")
        input_length = model_inputs.input_ids.shape[1]
        inputs = model_inputs.input_ids
    elif isinstance(batched_input, torch.Tensor):
        inputs = batched_input.to("cuda")
        input_length = batched_input.shape[1]
    else:
        logger.error(f"Unknown batched_input:{batched_input}")
        raise ValueError

    generated_ids = model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens, **kwargs)
    responses = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)
    # h2o clear
    if "llama" in pipeline_config['model_name'] or "longchat" in pipeline_config['model_name'] or "vicuna" in pipeline_config['model_name']:
        for name, m in model.named_modules():
            if isinstance(m, TAGET_MODULE["llama"]):
                m._reset_masks()
    elif "mistral" in pipeline_config['model_name']:
        for name, m in model.named_modules():
            if isinstance(m, TAGET_MODULE["mistral"]):
                m._reset_masks()
    else:
        logger.error(f"Unknown model architecture:{pipeline_config['model_name']}. Please add it target module architecture.")
        raise ValueError
    torch.cuda.empty_cache()

    return responses