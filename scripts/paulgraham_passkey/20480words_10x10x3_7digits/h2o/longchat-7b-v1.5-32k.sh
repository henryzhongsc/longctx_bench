compress_ratio=$1  # 2x, 4x, 6x, 8x
output_dir_root=$2

task="paulgraham_passkey"
dataset="20480words_10x10x3_7digits"
model="longchat-7b-v1.5-32k"
method="h2o"

python pipeline/${method}/main.py \
--exp_desc ${task}_${dataset}_${model}_${method}_${compress_ratio} \
--pipeline_config_dir config/pipeline_config/${method}/${task}/${model}/${compress_ratio}.json \
--eval_config_dir config/eval_config/${task}/${dataset}.json \
--output_folder_dir ${output_dir_root}/${task}/${dataset}/${method}/${model}/${compress_ratio}/
