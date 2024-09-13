output_dir_root=$1

task="longbench"
model="llama3-8b-instruct"
method="flexgen"
bits="4bit"

for dataset in qasper; do
    python pipeline/${method}/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method}_${bits} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${bits}/${model}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${bits}/${model}/${dataset}/
done
