import os
import json
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output/longbench/')
args = parser.parse_args()

DATA_PATH = args.output_dir
DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa","2wikimqa",
                    "musique","gov_report","qmsum","multi_news","trec","triviaqa","samsum",
                    "passage_retrieval_en", "lcc","repobench-p", "passage_count"]
MODELS = ['longchat-7b-v1.5-32k', 'mistral-7b-instruct-v0.2', 'llama3-8b-instruct']
METHODS = ['infllm', 'streamingllm']
COMPRESSION_RATES = ['2x','4x','6x','8x']

def get_table(compression_rate):
    df = pd.DataFrame(index=DATASETS, columns=MODELS)

    for method in METHODS:
        for dataset in DATASETS:
            for model in MODELS:
                file_name = os.path.join(DATA_PATH, method, model, compression_rate, dataset, 'raw_results.json')
                if os.path.isfile(file_name):
                    with open(file_name, 'r') as f:
                        results = json.load(f)
                        key = list(results.keys())[0]
                        val = results[key]
                        df.at[dataset, model] = val
            
        df.to_csv(os.path.join(DATA_PATH, f'{method}_{compression_rate}.csv'), index_label="Dataset/Model")


for rate in COMPRESSION_RATES:
    get_table(rate)
