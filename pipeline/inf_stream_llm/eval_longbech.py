import logging
import os
import json

logger = logging.getLogger("main")
import pandas as pd
from tqdm import tqdm

import infllm_utils
import inf_llm
import eval.longbench_utils.eval_long_bench as longbench_eval

def eval_longbech(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    data = longbench_eval.load_data(eval_params)
    raw_results = []

    if pipeline_params['method'] == 'inf-llm' or pipeline_params['method'] == 'stream-llm':
        logger.info(f'Starting LongBench evaluation via {pipeline_params["method"]}.')
        model, tokenizer = inf_llm.initialize_model_tokenizer(pipeline_config=pipeline_params)
        preds = infllm_utils.get_pred(
                                    model, tokenizer, data,
                                    eval_params, pipeline_params)

        out_path = os.path.join(config['management']['output_folder_dir'], 'pred')
        out_path = os.path.join(out_path, pipeline_params['method'])
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
        out_path = os.path.join(out_path, f'{eval_params["dataset"]}_{pipeline_params["chat_template"]}_compression_{pipeline_params["compression_ratio"]}.jsonl')

        # write predictions
        with open(out_path, "w", encoding="utf-8") as f:
            for pred, raw_config in zip(preds, data):
                json.dump(pred, f, ensure_ascii=False)
                f.write('\n')
                raw_config["response"] = pred
    else:
        logger.error(f"Invalid pipeline_params['method'] input: {pipeline_params['method']}.")
        raise ValueError

    processed_results, raw_results = longbench_eval.eval(pred_dir=config['management']['output_folder_dir'], model=pipeline_params['method'], eval_params=eval_params)

    return processed_results, raw_results
