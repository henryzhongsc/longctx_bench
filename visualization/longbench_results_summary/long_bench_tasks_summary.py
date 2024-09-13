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
TASK_DATASETS = {
    'single_doc_qa': ["narrativeqa", "qasper", "multifieldqa_en"],
    'multi_doc_qa': ["hotpotqa","2wikimqa", "musique"],
    'summarization': ["gov_report","qmsum","multi_news"],
    'few_shots': ["trec","triviaqa","samsum"],
    'synthetic': ["passage_retrieval_en"],
    'code': ["lcc","repobench-p"]
}    

def get_task_results():
    results = {}
    ind_dataset_result = {}
    task_ave_result = {}

    # Get individual dataset result
    for dataset in DATASETS:
        file_name = os.path.join(DATA_PATH, dataset, 'output_config.json')
        if os.path.isfile(file_name):
            with open(file_name, 'r') as f:
                result = json.load(f)
                result = result['eval_results']['processed_results']
                key = list(result.keys())[0]
                val = result[key]
                ind_dataset_result[dataset] = val
    
    results['individual_dataset_result'] = ind_dataset_result

    # Get task-average dataset result
    for task, datasets in TASK_DATASETS.items():
        task_ave_result[task] = 0
        for dataset in datasets:
            task_ave_result[task] += ind_dataset_result[dataset]
        task_ave_result[task] =  np.round(task_ave_result[task] / len(datasets), decimals = 2)

    results['task_average_result'] = task_ave_result

    #Get overall average result
    average_result = 0
    for dataset in DATASETS:
        if dataset != 'passage_count':
            average_result += ind_dataset_result[dataset]
    results['LB_average_result'] = np.round(average_result / (len(DATASETS) - 1), decimals = 2)

    # Save result
    output_result_path = os.path.join(DATA_PATH, 'longbench_result_summary.json')
    with open(output_result_path, "w+") as output_file:
        json.dump(results, output_file, indent = 4)
        print(f'Complete writing task summary to {output_result_path}')

get_task_results()
