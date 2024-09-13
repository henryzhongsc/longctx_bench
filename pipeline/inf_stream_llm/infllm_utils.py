import os
from tqdm import tqdm
from eval.longbench_utils.constants import LONGBENCH_DATASET
from pipeline.model_utils import build_chat
from inf_llm import patch_hf, GreedySearch, patch_model_center

def post_process(pred, chat_template, dataset):
    if chat_template == "qwen":
        pred = pred.split("<|im_end|>")[0]

    if dataset == "samsum":
        pred = pred.split("\n")[0].strip()

    return pred

def compress(eval_params, pipeline_params, tokenized_prompt, model, tokenizer):
    compression_ratio = pipeline_params.get('compression_ratio')
    init_ratio = pipeline_params.get('init_ratio')
    local_ratio = pipeline_params.get('local_ratio')
    context_ratio = pipeline_params.get('context_ratio')

    total_compressed_tokens = int(compression_ratio * min(len(tokenized_prompt), pipeline_params['model_max_len']))
    pipeline_params['n_init'] = max(int(total_compressed_tokens * init_ratio), 0)
    if context_ratio:
        pipeline_params['topk'] = max(0, int(total_compressed_tokens * context_ratio) // pipeline_params['block_size'])
        pipeline_params['max_cached_block'] = pipeline_params['topk']
        pipeline_params['n_local'] = max(pipeline_params['exc_block_size'], int(total_compressed_tokens - pipeline_params['n_init'] - pipeline_params['topk'] * pipeline_params['block_size']))
    else:
        pipeline_params['n_local'] = max(1, int(total_compressed_tokens - pipeline_params['n_init']))

    compressed_model = patch_hf(model, pipeline_params['method'], **pipeline_params)
    return GreedySearch(compressed_model, tokenizer)

def get_pred(
    model, tokenizer, data,
    eval_params, pipeline_params,
    truncation: str = None, rank: int = None, 
    world_size: int = None, verbose: bool = False
):
    preds = []
    data = list(data)

    if world_size is not None:
        data = data[rank::world_size]

    searcher = GreedySearch(model, tokenizer)
    cur = 0
    total = len(data)
    evaluating_longbench = eval_params['dataset'] in LONGBENCH_DATASET

    for json_obj in tqdm(data):
        if evaluating_longbench:
            prompt = eval_params['instruction'].format(**json_obj)
        else:
            prompt = json_obj

        extra_end_token_ids = []
        if pipeline_params['chat_template'] == "llama3":
            extra_end_token_ids.append(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])

        if pipeline_params['chat_template'] == "qwen":
            extra_end_token_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

        if eval_params['dataset'] == "samsum":
            extra_end_token_ids.append(tokenizer.encode("\n", add_special_tokens=False)[-1])

        if eval_params['dataset'] not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, pipeline_params['chat_template'])

            if pipeline_params['chat_template'].strip().lower() in ['mistral_instruct']:
                add_special_tokens = False
            else:
                add_special_tokens = True

        else:
            add_special_tokens = True

        # Truncation if necessary
        tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]
        if len(tokenized_prompt) > pipeline_params['model_max_len'] and pipeline_params.get('truncation_mode') == 'middle' and (not pipeline_params.get('out_of_max_len_allowed')):
            half = int(pipeline_params['model_max_len']/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=add_special_tokens).input_ids[0]

        if 'compression_ratio' in pipeline_params:
            searcher = compress(eval_params, pipeline_params,
                                tokenized_prompt, model, tokenizer)

        output = searcher.generate(
            input_ids = tokenized_prompt,
            max_length=eval_params['max_new_tokens'],
            chunk_size=pipeline_params.get('chunk_size'),
            extra_end_token_ids=extra_end_token_ids
        )

        pred = post_process(output[0], pipeline_params['chat_template'], eval_params['dataset'])
        if eval_params["dataset"] == "magic_city_number_retrieval" or eval_params["dataset"] == 'passkey_retrieval':
            preds.append({"pred": pred})
        else:
            preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"], "token_length": len(tokenized_prompt) + eval_params['max_new_tokens']})
        searcher.clear()
        cur += 1
        if verbose:
            logger.info(f"----------{cur}/{total}----------")
            logger.info("Length: ", len(tokenized_prompt))
            logger.info("Question:", prompt[-100:])
            logger.info("Pred:", pred)
            logger.info("Answer:", json_obj["answers"])
            logger.info("")

    return preds
