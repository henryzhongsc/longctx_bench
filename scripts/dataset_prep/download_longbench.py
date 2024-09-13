import argparse
import os
from datasets import load_dataset

save_dir = './dataset/longbench'

all_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                    "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                    "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

for dataset in all_datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
    data.save_to_disk(os.path.join(save_dir, dataset))