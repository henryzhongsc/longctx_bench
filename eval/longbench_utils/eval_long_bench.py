import os
import json
import argparse
import numpy as np

from datasets import load_dataset, load_from_disk
from .metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def load_data(config):
    data = load_from_disk(f'{config["dataset_path"]}/{config["dataset"]}')

    return data

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    raw_results = []
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
        raw_results.append({
            'answers': prediction, 
            'score': score
            })

    return round(100 * total_score / len(predictions), 2), raw_results

def eval(pred_dir, model, eval_params):
    scores = dict()
    path = os.path.join(pred_dir, "pred", f"{model}/")
    all_files = os.listdir(path)
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = eval_params['dataset']
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])

        score, raw_results = scorer(dataset, predictions, answers, all_classes)
        scores[dataset2metric[dataset].__name__] = score

    out_path = f"{pred_dir}/pred/{model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores, raw_results
