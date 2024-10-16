# KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches

[![Gul'dan_everything](https://github.com/henryzhongsc/longctx_bench/blob/main/visualization/plotted/Gul'dan_everything.png)](https://www.youtube.com/watch?v=TLzhlsEFcVQ&t=55s)

Jiayi Yuan\*, Hongyi Liu\*, Shaochen (Henry) Zhong\*, Yu-Neng Chuang, Songchen Li, Guanchu Wang, Duy Le, Hongye Jin, Vipin Chaudhary, Zhaozhuo Xu, Zirui Liu, Xia Hu

> This is the official implementation of our above-mentioned KV cache compression benchmark paper ([arXiv](https://arxiv.org/abs/2407.01527)). Work corresponds to Shaochen (Henry) Zhong and Zirui Liu.

---
## Changelog and To-do

* `2024/10/16`: All experiment logs, filtered, and raw results shared at [Google Drive link](https://drive.google.com/drive/folders/1PBB5bkR88QbaA1-dBTwP5xKLq8iEBcBb?usp=share_link).
* `2024/10/08`: v2 arXiv-ed.
  * [x] Update v2 to capture new experiment results under the current implementations.
  * [x] Add coverage of Mamba-2.
* `2024/09/19`: Accepted at EMNLP 2024 Findings (with a 4/5 meta!).
* `2024/07/01`: v1 arXiv-ed.
* `2024/06/15`: Initial submission to June ARR 2024.



---

## Overview

Token evictions, KV cache quantizations, hard prompt compression, RNNs, RNN-Transformer hybrids — plenty of efficiency approaches claim they are long context capable, but which can stand up to comprehensive scrutiny, and what are the trade-offs? Our benchmark paper attempts to answer such questions by evaluating many exemplar methods among the above schools against various long context tasks.

![cover_vis](https://github.com/henryzhongsc/longctx_bench/blob/main/visualization/plotted/cover_vis.png)

In this repo, we provide the implementation of all featured methods and evaluations. We intentionally make our codebase minimalistic for easier hacking and reproducing needs, yet we keep it extensible to include alternative or future-coming approaches that are not under our benchmark coverage. We plan to gradually incorporate more interesting methods and datasets into this repo.

---

## Environment Setup

We provide the minimum environment requirements to support the running of our project. This means there can be a slight difference depending on the actual automatic dependency-solving result of different systems. Given the efficiency focus of our benchmark, some of the featured methods require conflicting environments, causing us unable to provide a single unified environment for all featured methods. Thus, we try to control the version of some of the major packages (`torch`, `transformers`, etc.) and provide two sets of requirements.

* [`requirements/tf_4.40.txt`](https://github.com/henryzhongsc/longctx_bench/blob/main/requirements/tf_4.40.txt) supports the three transformers-based LLM baseline (`meta-llama/Meta-Llama-3-8B-Instruct`/`mistralai/Mistral-7B-Instruct-v0.2`/`lmsys/longchat-7b-v1.5-32k`), [Mamba](https://arxiv.org/abs/2312.00752), [Mamba-Chat](https://huggingface.co/havenhq/mamba-chat), [RecurrantGemma](https://storage.googleapis.com/deepmind-media/gemma/recurrentgemma-report.pdf), [RWKV-5-World](https://pypi.org/project/rwkv/), [FlexGen](https://arxiv.org/abs/2303.06865), [StreamingLLM](https://arxiv.org/abs/2309.17453), [InfLLM](https://arxiv.org/abs/2402.04617), [H2O](https://arxiv.org/abs/2306.14048), and [LLMLingua2](https://arxiv.org/abs/2403.12968).
* [`requirements/tf_4.36.txt`](https://github.com/henryzhongsc/longctx_bench/blob/main/requirements/tf_4.36.txt) supports [KIVI](https://arxiv.org/abs/2402.02750) (it does also run LLM baselines, FlexGen, StreamingLLM, and InfLLM; but we opt to conduct such experiments in the above environment for maximum possible consistency).
* [`requirements/tf_4.45.txt`](https://github.com/henryzhongsc/longctx_bench/blob/main/requirements/tf_4.45.txt) supports [Mamba 2](https://arxiv.org/abs/2405.21060). We note that with `Transfofmers v4.42+` there is [a bugfix](https://github.com/huggingface/transformers/pull/30536) regarding KV cache duplication thanks to [@Cyrilvallez](https://www.linkedin.com/posts/cyril-vallez-070a53220_today-is-my-first-day-at-hugging-face-i-activity-7241439500936036352-_aYL?utm_source=share&utm_medium=member_desktop), so please be vigilant when comparing memory measurements across different environments.

Should one be interested in reproducing a certain method, please look up the corresponding requirement file and install listed packages accordingly.

```
cd longctx_bench
pip install --upgrade pip
pip install -r requirements/<requirment_file_name>
```

We note that some packages often require installation later than other packages (e.g., `flash-attn` or custom kernel supports). In such cases, we commented out such "follow-up" packages in their requirement files so that users could manually install them later. Please carefully inspect the full requirement file and ensure all necessary packages are installed. As an example, suppose one wants to try out KIVI under our codebase; one should:

```
cd longctx_bench
pip install -r requirements/tf_4.36.txt
pip install flash-attn==2.6.3
cd pipeline/kivi/quant & pip install -e .
```

---

## Dataset and Access Preparation

Currently, our benchmark features 15 datasets (NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMQA, Musique, GovReport, QMSum, MultiNews, TREC, TriviaQA, SAMSum, PassageRetrieval, LCC, RepoBench-P) from [**LongBench**](https://github.com/THUDM/LongBench) and a [**Needle-in-a-Haystack/PaulGraham Passkey**](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) test with [Paul Graham essays](https://paulgraham.com/articles.html) as background and [passkey retrieval](https://arxiv.org/abs/2305.16300) as needle, following the spirit of [Arize-ai](https://github.com/Arize-ai/LLMTest_NeedleInAHaystack2)'s NIAH implementation utilized in Google DeepMind's [Gemini 1.5 report](https://arxiv.org/pdf/2403.05530).

The needle test's dataset are constructed on the fly, where all necessary material is already supplied in this repo. However, the LongBench datasets would require some extra preparations. We supply the following script to process LongBench:

```
python scripts/dataset_prep/download_longbench.py
```

Our paper features some models that are gated (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`). So please supply your HuggingFace access token under [`config/access_tokens.py`](https://github.com/henryzhongsc/longctx_bench/blob/main/config/access_tokens.py). You may consider setting `git update-index --skip-worktree config/access_tokens.py` so that Git will no longer track this file to avoid your locally stored token accidentally get synced to upstream.

---
## Experiment Reproduction

We supply all scripts in the [`scripts`](https://github.com/henryzhongsc/longctx_bench/blob/main/scripts) folder, with a folder structure that clearly indicate which script is for which experiment. E.g., should one want to verify the needle results of `mistral-7b-instruct-v0.2` model without any compression applied, one may do so via:

```
bash scripts/paulgraham_passkey/20480words_10x10x3_7digits/baseline/mistral-7b-instruct-v0.2.sh <output_dir_root>
```

Similarly, should one expect the LongBench results of `llama3-8b-instruct` model with KIVI-powered compression in 2bit, one may achieve such via:

```
bash scripts/longbench/kivi/2bit/llama3-8b-instruct.sh <output_dir_root>
```


> Given the long context tasks are often time-consuming to evaluate, we additionally supply a set of scripts in the [`scripts/quick_start`](https://github.com/henryzhongsc/longctx_bench/blob/main/scripts/quick_start) folder, which are quick to conclude (just Qasper for LongBench and just two of the longest needle inputs for PaulGraham Passkey). They are solely here as proxies to confirm the code and environment are (likely) running fine before committing to longer evaluations. We recommend trying these scripts first.

For viewer's convenience of comparison and potential table/plot construction needs, we share all experiment logs, filtered, and raw results at [Google Drive link](https://drive.google.com/drive/folders/1PBB5bkR88QbaA1-dBTwP5xKLq8iEBcBb?usp=share_link). The exact experiment results we reported can be find under the `result_only` folder, which are calculated base on the raw results shared under `full_output`. All experiments are done with respect to [commit 3e9d810](https://github.com/henryzhongsc/longctx_bench/commit/3e9d810f93cfdecc87d4538e1721d08047d66010).


---
## Result Digestion

Once an experiment is running, one may monitor real-time printouts in the terminal as well as the `exp.log` file under `<output_folder_dir>`. Once an experiment is concluded, the **final results can be find in a subfolder under `<output_dir_root>`.** The exact subfolder location varies depending on what experiment is being executed since our provided script would make a subfolder structure that is legible and makes sense for each method. Here is an example for InfLLM on PaulGraham Passkey; the results can be found under `<output_folder_dir>`.

```
compress_ratio=$1 # 2x, 4x, 6x, 8x
output_dir_root=$2

task="paulgraham_passkey"
dataset="20480words_10x10x3_7digits"
model="mistral-7b-instruct-v0.2_infinity_model_len"
method="infllm"

python pipeline/inf_stream_llm/main.py \
--exp_desc ${task}_${dataset}_${model}_${method}_${compress_ratio} \ # this argument is purely cosmetic.
--pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${compress_ratio}.json \
--eval_config_dir config/eval_config/${task}/${dataset}.json \
--output_folder_dir ${output_dir_root}/${task}/${dataset}/${method}/${model}/${compress_ratio}/
```

Under this `<output_folder_dir>` folder, one may expect the following components.

* `input_config` folder: This folder contains an `input_pipeline_config.json` and an`input_eval_config.json`. These are the carbon copy of the configs supplied to the `pipeline_config_dir` and `eval_config_dir` arguments of the script. Such configs are copied here for easy replication purposes as these two configs basically define an experiment.
* `output_config.json`: This file provides a fuse of the above two input configs and some management information (e.g., start/end time of a job). **Most importantly, it highlights the main reported metrics under the key `eval_results`.**
    * In the case of PaulGraham Passkey, it includes the accuracy of each depth-length combination, the average accuracy at each length (`background_len_wise_results`), and the overall accuracy (`overall_results`) in both exact and partial match fashion.
    * **If you are only interested in comparing metrics reported in the main tables of our paper, this is the only place you need to check out.** We put such results alongside the two input configs so there is no chance of attributing the results to a wrong setting.
* `raw_results.json`: This file registers the fine-grain results of the concluded experiment, even if they are not reported in our paper (e.g., individual scoring of each output). It also registers all newly generated tokens upon each input for monitoring/debugging purposes.
* `exp.log`: This is a carbon copy of the real-time printouts to the terminal.

> For LongBench, we treat each of its subdatasets as an individual eval and loop them over, as demoed [here](https://github.com/henryzhongsc/longctx_bench/blob/main/scripts/longbench/baseline/llama3-8b-instruct.sh). In this case, we additionally channel the [LongBench summarization scripts](https://github.com/henryzhongsc/longctx_bench/blob/main/visualization/longbench_results_summary) to summarize all results corresponding to a full LongBench run into one json file at the subdataset level.

---
## Codebase Design and Contribution

Our codebase mainly revolves around the two configs — [`pipeline_config`](https://github.com/henryzhongsc/longctx_bench/blob/main/config/pipeline_config) and [`eval_config`](https://github.com/henryzhongsc/longctx_bench/blob/main/config/eval_config) — where the former one defines a method, and the latter one defines an evaluation scheme. Their code implementations can be respectively found under the [`pipeline`](https://github.com/henryzhongsc/longctx_bench/blob/main/pipeline) and [`eval`](https://github.com/henryzhongsc/longctx_bench/blob/main/eval) folders. We keep the eval implementations "singleton" for each dataset to ensure they are fairly tested among different pipelines. Yet, for pipeline implementation, we intentionally repeat some code components (e.g., the `pipeline/<method>/eval_longbench.py` and `eval_passkey_retrieval.py` files are largely the same under different methods) as we want to keep each pipeline/method folder relatively self-contained for better readability; and to make everything loosely coupled to support intervention requests coming at different angles.

Should you want to add a new evaluation, you may consider adding an `eval/<new_eval_utils>` folder for its eval utilities, and supply corresponding `eval_<new_eval>.py` files under all `pipeline/<method>` folders you'd care to evaluate. Should you want to add a new method to our codebase, please consider adding a `pipeline/<method>` folder and couple with the evaluations you'd care to conduct with necessary `eval_<a_certain_eval>.py` files.

---


Should you need to refer to this work or find our codebase useful, please consider citing our work as:


```
@inproceedings{yuan_liu_zhong_2024_kvcache_comp_benchmnark,
    title={KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches},
    author={Jiayi Yuan and Hongyi Liu and Shaochen Zhong and Yu-Neng Chuang and Songchen Li and Guanchu Wang and Duy Le and Hongye Jin and Vipin Chaudhary and Zhaozhuo Xu and Zirui Liu and Xia Hu},
    booktitle={The 2024 Conference on Empirical Methods in Natural Language Processing},
    year={2024},
}
```
