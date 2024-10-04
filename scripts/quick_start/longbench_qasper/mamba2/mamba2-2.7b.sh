output_dir_root=$1

task="longbench"
model="mamba2-2.7b"
method="mamba2"

for dataset in qasper; do
    python pipeline/${method}/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${model}/${dataset}/
done
