import logging
import os
import json

logger = logging.getLogger("main")
import pandas as pd
from tqdm import tqdm

import eval.longbench_utils.eval_long_bench as longbench_eval

import torch
from transformers import LlamaConfig, MistralConfig, AutoTokenizer
from llmlingua import PromptCompressor

from pipeline.model_utils import build_chat
from eval.longbench_utils.eval_long_bench import load_data
from llmlingua import PromptCompressor
from inference import initialize_model_tokenizer, prompt_compressor, batch_generate

def get_pred(model, tokenizer, data, device, pipeline_params, eval_params):
    preds = []
    for json_obj in tqdm(data):
        prompt = eval_params['instruction'].format(**json_obj)

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > pipeline_params['model_max_len'] and pipeline_params.get('truncation_mode') == 'middle' and (not pipeline_params.get('out_of_max_len_allowed')):
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

        pred = batch_generate(input.input_ids, model, tokenizer, eval_params['max_new_tokens'], **kwargs)[0]
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})

    return preds

def compress_data(data, pipeline_params):
    compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                    use_llmlingua2=True
    )
    compression_key = "context" if "context" in data.features.keys() else "input"
    cp_pt_list, cp_pt_len_list = prompt_compressor(compressor, data[compression_key], rate=float(1/pipeline_params["cp_rate"]))
    data = data.remove_columns(compression_key).add_column(compression_key, cp_pt_list)
    data = data.remove_columns("length").add_column("length", cp_pt_len_list)

    return data

def eval_longbench(config):
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']
    data = load_data(eval_params)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    raw_results = []

    assert pipeline_params['method'] == 'llmlingua'
    logger.info(f'Starting LongBench evaluation via {pipeline_params["method"]}.')

    model, tokenizer = initialize_model_tokenizer(pipeline_params)
    model.eval()

    # Strat compression
    data = compress_data(data, pipeline_params)

    # Get prediction with compressed prompts
    preds= get_pred(
                    model, tokenizer, data,
                    device, pipeline_params, eval_params
                )

    out_path = os.path.join(config['management']['output_folder_dir'], 'pred')
    out_path = os.path.join(out_path, pipeline_params['method'])
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_path = os.path.join(out_path, f'{eval_params["dataset"]}_{pipeline_params["chat_template"]}_compression_{pipeline_params["cp_rate"]}.jsonl')

    # write predictions
    with open(out_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write('\n')

    processed_results, raw_results = longbench_eval.eval(pred_dir=config['management']['output_folder_dir'], model=pipeline_params['method'], eval_params=eval_params)

    return processed_results, raw_results
