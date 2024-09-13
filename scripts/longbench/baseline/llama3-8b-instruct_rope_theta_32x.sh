output_dir_root=$1

task="longbench"
model="llama3-8b-instruct_rope_theta_32x"
method="baseline"

for dataset in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_retrieval_en passage_count lcc repobench-p; do
    python pipeline/${method}/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${dataset}/
done

python visualization/longbench_results_summary/long_bench_tasks_summary.py \
    --output_dir ${output_dir_root}/${task}/${method}/${model}/
