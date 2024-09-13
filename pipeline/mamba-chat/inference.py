import logging
logger = logging.getLogger("main")

import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import  MambaConfig, MambaForCausalLM,AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

def initialize_model_tokenizer(pipeline_config):
    tokenizer = AutoTokenizer.from_pretrained(pipeline_config['model_name']) # "state-spaces/mamba-2.8b-hf"
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token
    model = MambaLMHeadModel.from_pretrained(
        pretrained_model_name=pipeline_config['model_name'],
        dtype=torch.float16,
        device="cuda",
    )
    return model, tokenizer



def batch_generate(batched_input, model, tokenizer, max_new_tokens, pipeline_config, **kwargs):
    model.eval()

    if isinstance(batched_input[0], str):
        model_inputs = tokenizer(
            batched_input, return_tensors="pt", padding=True
        ).to("cuda")
        input_length = model_inputs.input_ids.shape[1]
        inputs = model_inputs.input_ids
    elif isinstance(batched_input, torch.Tensor):
        inputs = batched_input.to("cuda")
        input_length = batched_input.shape[1]
    else:
        logger.error(f"Unknown batched_input:{batched_input}")
        raise ValueError


    generated_ids = model.generate(inputs, top_p = 0, temperature = 1, max_length=max_new_tokens, **kwargs)
    responses = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return responses
