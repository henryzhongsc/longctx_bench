import torch
from .utils import patch_hf, GreedySearch, patch_model_center
from transformers import LlamaConfig, MistralConfig, AutoTokenizer, LlamaForCausalLM, MistralForCausalLM

def initialize_model_tokenizer(pipeline_config):
    dtype = torch.float16
    model = None
    tokenizer = None

    if 'llama' in pipeline_config['model_name'].lower() or 'longchat' in pipeline_config['model_name'].lower():
        config = LlamaConfig.from_pretrained(pipeline_config['model_name'])
        if 'llama' in pipeline_config['model_name'].lower():
            config.rope_theta *= pipeline_config['rope_theta_factor']
        tokenizer = AutoTokenizer.from_pretrained(pipeline_config['tokenizer_name'])
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pipeline_config['model_name'],
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True,
            device_map="auto",
        )

    elif 'mistral' in pipeline_config['model_name'].lower():
        config = MistralConfig.from_pretrained(pipeline_config['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(pipeline_config['tokenizer_name'],
                                            use_fast=False,
                                            trust_remote_code=True)
        model = MistralForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pipeline_config['model_name'],
                config=config,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                use_flash_attention_2=True,
                device_map="auto",
            )
    return model, tokenizer
