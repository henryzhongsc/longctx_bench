output_dir_root=$1

task="longbench"
model="mistral-7b-instruct-v0.2"
method="kivi"
bits="2bit"

for dataset in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_retrieval_en passage_count lcc repobench-p; do
    python pipeline/${method}/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method}_${bits} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${bits}/${model}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${bits}/${model}/${dataset}/
done

python visualization/longbench_results_summary/long_bench_tasks_summary.py \
    --output_dir ${output_dir_root}/${task}/${method}/${bits}/${model}/
