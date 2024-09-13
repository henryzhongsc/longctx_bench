import sys
import os
import json
import datetime
import pdb
import torch
from zoneinfo import ZoneInfo

current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(base_dir)
os.chdir(base_dir)

from config.access_tokens import hf_access_token
from huggingface_hub import login
login(token=hf_access_token)

from eval.longbench_utils.constants import LONGBENCH_E_DATASET, LONGBENCH_DATASET
import pipeline.main_utils as main_utils
from eval_passkey_retrieval import eval_passkey_retrieval
from eval_longbench import eval_longbench


SEED = 42
main_utils.lock_seed(SEED)
torch.cuda.reset_peak_memory_stats()

ct_timezone = ZoneInfo("America/Chicago")
start_time = datetime.datetime.now(ct_timezone)
args = main_utils.parse_args()
config = main_utils.register_args_and_configs(args)
logger = main_utils.set_logger(args.output_folder_dir, args)


logger.info(f"Experiment {config['management']['exp_desc']} (SEED={SEED}) started at {start_time} with the following config: ")
logger.info(json.dumps(config, indent=4))



if config['eval_params']['dataset'] == 'passkey_retrieval' or config['eval_params']['dataset'] == 'magic_city_number_retrieval':
    processed_results, raw_results = eval_passkey_retrieval(config)
    main_utils.register_result(processed_results, raw_results, config)

elif config['eval_params']['dataset'] in LONGBENCH_DATASET or config['eval_params']['dataset'] in LONGBENCH_E_DATASET:
    processed_results, raw_results = eval_longbench(config)
    main_utils.register_result(processed_results, raw_results, config)

else:
    logger.error(f"Invalid config['eval_params']['dataset'] input: {config['eval_params']['dataset']}.")
    raise ValueError



end_time = datetime.datetime.now(ct_timezone)
main_utils.register_exp_time(start_time, end_time, config)
main_utils.register_output_config(config)
logger.info(f"Experiment {config['management']['exp_desc']} ended at {end_time}. Duration: {config['management']['exp_duration']}")
