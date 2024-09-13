import logging
logger = logging.getLogger("main")

import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM

def kivi_params_check(pipeline_params):
    assert pipeline_params['method'] == 'KIVI'
    assert pipeline_params['k_bits'] == 2 or pipeline_params['k_bits'] == 4
    assert pipeline_params['v_bits'] == 2 or pipeline_params['v_bits'] == 4
    assert pipeline_params['residual_length'] >= pipeline_params['group_size']


def initialize_model_tokenizer(pipeline_params):
    kivi_params_check(pipeline_params)

    config = AutoConfig.from_pretrained(pipeline_params['model_name'])
    if 'rope_theta_factor' in pipeline_params and hasattr(config, 'rope_theta'):
        config.rope_theta *= pipeline_params['rope_theta_factor']

    config.use_flash = pipeline_params['use_flash_attn']
    config.k_bits = pipeline_params['k_bits']
    config.v_bits = pipeline_params['v_bits']
    config.group_size = pipeline_params['group_size']
    config.residual_length = pipeline_params['residual_length']
    dtype = torch.float16


    if 'llama' in pipeline_params['model_name'].lower() or 'longchat' in pipeline_params['model_name'].lower():
        from models.llama_kivi import LlamaForCausalLM_KIVI
        model = LlamaForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=pipeline_params['model_name'],
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto")
    elif 'mistral' in pipeline_params['model_name'].lower():
        from models.mistral_kivi import MistralForCausalLM_KIVI
        model = MistralForCausalLM_KIVI.from_pretrained(
            pretrained_model_name_or_path=pipeline_params['model_name'],
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto")
    else:
        logger.error(f"Unsupported pipeline_params['model_name']: {pipeline_params['model_name']}")
        raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained(pipeline_params['tokenizer_name'], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')
    return model, tokenizer


def batch_generate(batched_input, model, tokenizer, max_new_tokens, **kwargs):
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

    torch.cuda.empty_cache()

    return responses
