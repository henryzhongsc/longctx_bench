import logging
logger = logging.getLogger("main")

import argparse
import sys
import os
import copy
import json
import datetime
from zoneinfo import ZoneInfo
import random

import torch
import numpy as np
import transformers



def lock_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_desc', type=str, help='experiment description.')
    parser.add_argument('--pipeline_config_dir', type=str, help='file path of pipeline config.')
    parser.add_argument('--eval_config_dir', type=str, help='file path of eval config.')
    parser.add_argument('--output_folder_dir', default='', type=str, help='path of output model')
    parser.add_argument('--job_post_via', default='terminal', type=str, help='slurm_sbatch')    
    # parser.add_argument("--language_model_path", type=str)
    # parser.add_argument("--tokenizer_name", type=str)
    # parser.add_argument("--context_window", type=int, default=3900)
    # parser.add_argument("--max_new_tokens", type=int, default=256)
    # parser.add_argument("--n_gpu_layers", type=int)
    args = parser.parse_args()

    if args.output_folder_dir != '':
        if args.output_folder_dir[-1] != '/':
            args.output_folder_dir  += '/'
    else:
        logger.error(f'Valid {args.output_folder_dir} is required.')

    return args


# Output in terminal and exp.log file under output_folder_dir.
def set_logger(output_folder_dir, args): 
    ct_timezone = ZoneInfo("America/Chicago")
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s : %(message)s")    
    log_formatter.converter = lambda *args: datetime.datetime.now(ct_timezone).timetuple()
    file_handler = logging.FileHandler(output_folder_dir + 'exp.log', mode = 'w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    return logger


def register_args_and_configs(args):

    # Make outer output dir.
    if not os.path.isdir(args.output_folder_dir):
        os.makedirs(args.output_folder_dir)
        logger.info(f'Output folder dir {args.output_folder_dir} created.')
    else:
        logger.info(f'Output folder dir {args.output_folder_dir} already exist.')


    # Copy input eval config to output dir.
    with open(args.eval_config_dir) as eval_config_f:
        eval_config = json.load(eval_config_f)
        logger.info(f'Input eval config file {args.eval_config_dir} loaded.')
    
    
    # Make subdir under output dir to store input configs.
    input_config_subdir = eval_config['management']['sub_dir']['input_config']
    if not os.path.isdir(args.output_folder_dir + input_config_subdir):
        os.makedirs(args.output_folder_dir + input_config_subdir)
        logger.info(f'Input config subdir {args.output_folder_dir + input_config_subdir} created.')
    else:
        logger.info(f'Input config subdir {args.output_folder_dir + input_config_subdir} already exist.')

    input_eval_config_path = args.output_folder_dir + input_config_subdir + 'input_eval_config.json'
    with open(input_eval_config_path, "w+") as input_eval_config_f:
        json.dump(eval_config, input_eval_config_f, indent = 4)
        logger.info(f'Input eval config file {args.eval_config_dir} saved to {input_eval_config_path}.')

    # Copy input pipeline config to output dir.
    with open(args.pipeline_config_dir) as pipeline_config_f:
        pipeline_config = json.load(pipeline_config_f)
        logger.info(f'Input pipeline config file {args.pipeline_config_dir} loaded.')

    input_pipeline_config_path = args.output_folder_dir + input_config_subdir + 'input_pipeline_config.json'
    with open(input_pipeline_config_path, "w+") as input_pipeline_config_f:
        json.dump(pipeline_config, input_pipeline_config_f, indent = 4)
        logger.info(f'Input pipeline config file {args.pipeline_config_dir} saved to {input_pipeline_config_path}.')


    # Fuse and complete pipeline config, eval config, and args from argparser into a general config.
    config = dict()
    config['pipeline_params'] = pipeline_config['pipeline_params']
    config['eval_params'] = eval_config['eval_params']
    config['eval_results'] = dict() # processed result

    config['management'] = dict()
    config['management']['exp_desc'] = args.exp_desc
    config['management']['pipeline_config_dir'] = args.pipeline_config_dir
    config['management']['eval_config_dir'] = args.eval_config_dir
    config['management']['output_folder_dir'] = args.output_folder_dir
    config['management']['job_post_via'] = args.job_post_via
    if config['management']['job_post_via'] == 'slurm_sbatch':     # Add slurm info to config['management'] if the job is triggered via slurm sbatch.
        config['management']['slurm_info'] = register_slurm_sbatch_info()
    config['management']['sub_dir'] = eval_config['management']['sub_dir']

    return config


def register_slurm_sbatch_info():
    slurm_job_id = os.environ['SLURM_JOB_ID']
    slurm_job_name = os.getenv('SLURM_JOB_NAME')
    slurm_out_file_dir = os.getenv('SLURM_SUBMIT_DIR') + '/slurm-' + os.getenv('SLURM_JOB_ID') + '.out'

    logger.info(f'Slurm job #{slurm_job_id} ({slurm_job_name}) running with slurm.out file at {slurm_out_file_dir}.')

    return {"slurm_job_id": slurm_job_id, "slurm_job_name": slurm_job_name, "slurm_out_file_dir": slurm_out_file_dir}



def register_result(processed_results, raw_results, config):
    
    raw_results_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['raw_results']
    with open(raw_results_path, "w+") as raw_results_f:
        json.dump(raw_results, raw_results_f, indent = 4)
        logger.info(f'raw_results file saved to {raw_results_path}.')


    config['eval_results']['processed_results'] = processed_results
    logger.info('Experiments concluded, below is the raw_results: ')
    logger.info(json.dumps(raw_results, indent=4))

    logger.info('##### And below is the processed_results: #####')
    logger.info(json.dumps(config['eval_results']['processed_results'], indent=4))


def register_exp_time(start_time, end_time, config):
    config['management']['start_time'] = str(start_time)
    config['management']['end_time'] = str(end_time)
    config['management']['exp_duration'] = str(end_time - start_time)


def register_output_config(config):
    output_config_path = config['management']['output_folder_dir'] + config['management']['sub_dir']['output_config']
    with open(output_config_path, "w+") as output_config_f:
        json.dump(config, output_config_f, indent = 4)
        logger.info(f'output_config file saved to {output_config_path}.')