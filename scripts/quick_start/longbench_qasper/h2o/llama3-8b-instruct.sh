compress_ratio=$1 # 2x, 4x, 6x, 8x
output_dir_root=$2

task="longbench"
model="llama3-8b-instruct"
method="h2o"

for dataset in qasper; do
    python pipeline/${method}/main.py \
    --exp_desc ${task}_${dataset}_${model}_${method}_${compress_ratio} \
    --pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${compress_ratio}.json \
    --eval_config_dir config/eval_config/${task}/${dataset}.json \
    --output_folder_dir ${output_dir_root}/${task}/${method}/${compress_ratio}/${model}/${dataset}/
done
