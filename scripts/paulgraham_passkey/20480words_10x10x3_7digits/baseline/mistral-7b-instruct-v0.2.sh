output_dir_root=$1

task="paulgraham_passkey"
dataset="20480words_10x10x3_7digits"
model="mistral-7b-instruct-v0.2"
method="baseline"

python pipeline/${method}/main.py \
--exp_desc ${task}_${dataset}_${model}_${method} \
--pipeline_config_dir config/pipeline_config/${method}/${task}/${model}.json \
--eval_config_dir config/eval_config/${task}/${dataset}.json \
--output_folder_dir ${output_dir_root}/${task}/${dataset}/${method}/${model}/
