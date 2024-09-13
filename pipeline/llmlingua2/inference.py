import logging
logger = logging.getLogger("main")

import torch
from tqdm import tqdm

from transformers import LlamaConfig, MistralConfig, AutoTokenizer, LlamaForCausalLM, MistralForCausalLM
from llmlingua import PromptCompressor

def initialize_model_tokenizer(pipeline_params):
    if 'llama' in pipeline_params['model_name'].lower() or 'longchat' in pipeline_params['model_name'].lower():
        model_config = LlamaConfig.from_pretrained(pipeline_params['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(pipeline_params['tokenizer_name'])

        model_config.use_flash = True

        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pipeline_params['model_name'],
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    elif 'mistral' in pipeline_params['model_name'].lower():
        model_config = MistralConfig.from_pretrained(pipeline_params['model_name'])
        tokenizer = AutoTokenizer.from_pretrained(
                pipeline_params['tokenizer_name'], 
                use_fast=False, 
                trust_remote_code=True
        )
        model_config.use_flash = True
            
        model = MistralForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pipeline_params['model_name'],
            config=model_config,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
    else:
        raise NotImplementedError

    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')

    return model, tokenizer

def prompt_compressor(compressor, original_prompt, rate):
    cp_pt_list = []
    cp_pt_len_list = []
    for prompt in tqdm(original_prompt):
        results = compressor.compress_prompt_llmlingua2(
            prompt,
            rate=rate,
            force_tokens=['\n', '.', '!', '?', ','],
            chunk_end_tokens=['.', '\n'],
            return_word_label=True,
            drop_consecutive=True
        )
        cp_pt_list.append(results['compressed_prompt'])
        cp_pt_len_list.append(results['compressed_tokens'])

    return cp_pt_list, cp_pt_len_list


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
