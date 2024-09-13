compress_ratio=$1 # 2x, 4x, 6x, 8x
output_dir_root=$2

task="longbench"
model="mistral-7b-instruct-v0.2_infinity_model_len"
method="infllm"

for dataset in narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_retrieval_en passage_count lcc repobench-p; do
    python pipeline/inf_stream_llm/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method}_${compress_ratio} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${compress_ratio}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${compress_ratio}/${model}/${dataset}/
done

python visualization/longbench_results_summary/long_bench_tasks_summary.py \
    --output_dir ${output_dir_root}/${task}/${method}/${compress_ratio}/${model}/
