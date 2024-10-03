import logging
logger = logging.getLogger("main")

import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM


def initialize_model_tokenizer(pipeline_params):
    config = AutoConfig.from_pretrained(pipeline_params['model_name'])
    if 'rope_theta_factor' in pipeline_params and hasattr(config, 'rope_theta'):
        config.rope_theta *= pipeline_params['rope_theta_factor']

    if pipeline_params['use_flash_attn']:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'eager'

    if 'mamba2' or 'mamba-codestral' in pipeline_params['model_name'].lower():
        from transformers import Mamba2Config, Mamba2ForCausalLM
        model = Mamba2ForCausalLM.from_pretrained(pipeline_params['model_name']).to("cuda")
    elif 'mamba' in pipeline_params['model_name'].lower():
        from transformers import MambaConfig, MambaForCausalLM
        model = MambaForCausalLM.from_pretrained(pipeline_params['model_name']).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(pipeline_params['model_name'], config=config, device_map="auto", attn_implementation=attn_implementation, torch_dtype=torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(pipeline_params['tokenizer_name'], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')
    return model, tokenizer


def batch_generate(batched_input, model, tokenizer, max_new_tokens):
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

    generated_ids = model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens)
    responses = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

    torch.cuda.empty_cache()

    return responses
