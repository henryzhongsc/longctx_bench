import logging
import os
import json

logger = logging.getLogger("main")
from tqdm import tqdm

from transformers import LlamaConfig, MistralConfig, AutoModelForCausalLM, AutoTokenizer
from llmlingua import PromptCompressor

import eval.passkey_utils.passkey_main as passkey_main
import eval.passkey_utils.passkey_utils as passkey_utils
from inference import initialize_model_tokenizer, batch_generate, prompt_compressor

def eval_passkey_retrieval(config):
    raw_exp_results = passkey_main.prepare_passkey_retrieval_input(config)
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    assert pipeline_params['method'] == 'llmlingua'

    logger.info('Starting evaluation via Llmlingua')

    model, tokenizer = initialize_model_tokenizer(pipeline_params)
    compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True
    )

    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    longest_input = raw_exp_results[-1]['full_input']
    passkey_utils.check_if_out_of_context_window(longest_input=longest_input,
                                                     model_max_len=pipeline_params['model_max_len'],
                                                     tokenizer=tokenizer,
                                                     out_of_max_len_allowed=pipeline_params['out_of_max_len_allowed'])

    batch_size = config['pipeline_params']['batch_size']

    # Suppose batch_size = 2, batched_raw_exp_results = [ [{}, {}], [{}, {}], ... ] where one {} is an element of raw_exp_log.
    batched_raw_exp_results = [raw_exp_results[i:i + batch_size] for i in
                                range(0, len(raw_exp_results), batch_size)]

    for i, one_batch in enumerate(batched_raw_exp_results):
        batched_input = [i['full_input'] for i in one_batch]
        batched_input, _ = prompt_compressor(compressor, batched_input, rate=float(1/pipeline_params["cp_rate"]))

        batched_responses = batch_generate(
                        batched_input=batched_input,
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=config['eval_params']['max_new_tokens']
        )

        for one_exp_results, one_response in zip(one_batch, batched_responses):
            one_exp_results['response'] = one_response

        logger.info(f'Finished evaluating batch {i + 1}/{len(batched_raw_exp_results)} (batch_size = {batch_size}).')
    logger.info(f'Finished evaluating all {len(batched_raw_exp_results)} batches (batch_size = {batch_size}).')

    processed_results, raw_results = passkey_utils.process_raw_exp_results(raw_exp_results=raw_exp_results, metrics=eval_params['eval_metrics'])
    logger.info('raw_exp_results processed.')

    return processed_results, raw_results
