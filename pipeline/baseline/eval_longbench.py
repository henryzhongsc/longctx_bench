import logging
import os
import json
import pdb

logger = logging.getLogger("main")
import pandas as pd
from tqdm import tqdm

import eval.longbench_utils.eval_long_bench as longbench_eval

import torch
import inference as inference
from pipeline.model_utils import build_chat
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval.longbench_utils.eval_long_bench import load_data


def get_pred(model, tokenizer, data, device, pipeline_params, eval_params):
    preds = []
    for json_obj in tqdm(data):
        prompt = eval_params['instruction'].format(**json_obj)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if 'truncation_mode' in pipeline_params and pipeline_params['truncation_mode'] == 'middle' and len(tokenized_prompt) > pipeline_params['model_max_len']:
            half = int(pipeline_params['model_max_len']/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

        if eval_params['dataset'] not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]: # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, pipeline_params['chat_template'])

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]
        kwargs = {
            "eos_token_id" : [tokenizer.eos_token_id]
        }
        if eval_params['dataset'] == "samsum": # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            kwargs.update({
                "min_length" : context_length+1,
                "eos_token_id" : [tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]]
            })

        pred = inference.batch_generate(input.input_ids, model, tokenizer, eval_params['max_new_tokens'], **kwargs)[0]
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})

    return preds

def eval_longbench(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    data = load_data(eval_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_results = []

    model, tokenizer = inference.initialize_model_tokenizer(pipeline_params=config['pipeline_params'])
    model.eval()
    preds= get_pred(model, tokenizer, data, device, pipeline_params, eval_params)

    out_path = os.path.join(config['management']['output_folder_dir'], 'pred')
    out_path = os.path.join(out_path, pipeline_params['method'])
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_path = os.path.join(out_path, f'{eval_params["dataset"]}_{pipeline_params["chat_template"]}.jsonl')

    # write predictions
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')

    processed_results, raw_results = longbench_eval.eval(pred_dir=config['management']['output_folder_dir'], model=pipeline_params['method'], eval_params=eval_params)

    return processed_results, raw_results
