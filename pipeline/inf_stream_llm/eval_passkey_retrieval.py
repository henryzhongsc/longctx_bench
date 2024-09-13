import logging
import os
import json

logger = logging.getLogger("main")
import pandas as pd
from tqdm import tqdm

import eval.passkey_utils.passkey_main as passkey_main
import eval.passkey_utils.passkey_utils as passkey_utils
import infllm_utils


import inf_llm as inf_llm

def eval_passkey_retrieval(config):
    raw_exp_results = passkey_main.prepare_passkey_retrieval_input(config)
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']

    if pipeline_params['method'] == 'inf-llm' or pipeline_params['method'] == 'stream-llm':
        logger.info(f'Starting evaluation via {pipeline_params["method"]}.')
        model, tokenizer = inf_llm.initialize_model_tokenizer(pipeline_config=pipeline_params)

        longest_input = raw_exp_results[-1]['full_input']

        passkey_utils.check_if_out_of_context_window(longest_input=longest_input,
                                                     model_max_len=pipeline_params['model_max_len'],
                                                     tokenizer=tokenizer,
                                                     out_of_max_len_allowed=pipeline_params['out_of_max_len_allowed'])

        data = [raw_exp['full_input'] for raw_exp in raw_exp_results]
                                                        
        preds = infllm_utils.get_pred(
                                        model, tokenizer, data,
                                        eval_params, pipeline_params)

        for raw_exp, pred in zip(raw_exp_results, preds):
            raw_exp['response'] = pred['pred']
        
        logger.info(f'Finished evaluating {pipeline_params["method"]} on passkey retrieval.')
    else:
        logger.error(f"Invalid pipeline_params['method'] input: {pipeline_params['method']}.")
        raise ValueError

    processed_results, raw_results = passkey_utils.process_raw_exp_results(raw_exp_results=raw_exp_results, metrics=eval_params['eval_metrics'])
    # logger.info(f"raw results is {raw_results}")
    logger.info('raw_exp_results processed.')
    # logger.info(raw_exp_results)

    return processed_results, raw_results