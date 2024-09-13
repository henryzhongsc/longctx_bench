import logging
logger = logging.getLogger("main")

import os.path

import torch

from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from rwkv.utils import PIPELINE_ARGS


def initialize_model_tokenizer(pipeline_config):

    title = "RWKV-5-World-7B-v2-20240128-ctx4096"
    model_path = hf_hub_download(repo_id="BlinkDL/rwkv-5-world", filename=f"{title}.pth")
    # model = RWKV(model=model_path, strategy='cuda fp16i8 *8 -> cuda fp16').cuda()
    model = RWKV(model=model_path, strategy='cuda fp16-> cuda fp16').cuda()
    tokenizer = PIPELINE(model, "rwkv_vocab_v20230424") # this is a pipeline
    return model, tokenizer


def batch_generate(batched_input, model, pipeline, max_new_tokens, pipeline_config):
    # model.eval()
    # import ipdb; ipdb.set_trace()
    # modify for rwkv pipeline tokenizer
    # if isinstance(batched_input[0], str):
    #     model_inputs = pipeline.encode(
    #         batched_input[0]
    #     )
    #     input_length = len(model_inputs)
    #     inputs = model_inputs
    # elif isinstance(batched_input, torch.Tensor):
    #     inputs = batched_input.to("cuda")
    #     input_length = batched_input.shape[1]
    # elif isinstance(batched_input, list): # RWKV decoding format, already decoded
    #     inputs = batched_input
    # else:
    #     logger.error(f"Unknown batched_input:{batched_input}")
    #     raise ValueError

    # inference loop
    # https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/utils.py
    # all_tokens = []
    # out_last = 0
    # token_count = max_new_tokens #500
    # temperature = 1
    # top_p = 0
    # out_str = ''
    # occurrence = {}
    # state = None
    # for i in range(token_count):
    #     out, state = model.forward(inputs if i == 0 else [token], state)
    #     for n in occurrence:
    #         out[n] -= (0.1 + occurrence[n] * 0.1)
    #     token = tokenizer.sample_logits(out, temperature=temperature, top_p=top_p)
    #     if token in [0]:
    #         break
    #     all_tokens += [token]
    #     for xxx in occurrence:
    #         occurrence[xxx] *= 0.996
    #     if token not in occurrence:
    #         occurrence[token] = 1
    #     else:
    #         occurrence[token] += 1
    #     tmp = tokenizer.decode(all_tokens[out_last:])
    #     if '\ufffd' not in tmp:
    #         out_str += tmp
    #         out_last = i + 1
    # responses = [out_str]

    if isinstance(batched_input, str):
        pass
    if isinstance(batched_input, list): # batch size = 1
        batched_input = batched_input[0]

    args = PIPELINE_ARGS(top_p= 0)
    # import ipdb; ipdb.set_trace()
    responses = pipeline.generate(batched_input, max_new_tokens, args)

    torch.cuda.empty_cache()
    return [responses]
