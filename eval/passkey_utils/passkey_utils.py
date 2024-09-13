import logging
logger = logging.getLogger("main")

import os
import re
import random
import copy
from itertools import cycle

import eval.passkey_utils.eval_metrics as eval_metrics



def get_intermediate_values_within_min_max(value_min, value_max, n_values, desc, int_rounding = False):
    if value_min == value_max:
        values = [value_min]
    elif value_min < value_max:
        n_value_intervals = n_values - 1 # Suppose working on depths, number of intervals between n_depths.
        values = [i * (value_max - value_min)/n_value_intervals + value_min for i in range(1, n_value_intervals)]
        values = [value_min] + values + [value_max]
        # suppose depth_min = 0, depth_max = 1, depth_num_intervals = 4. depths = [0, 0.33..., 0.66..., 1]
        # suppose depth_min = 0.25, depth_max = 0.75, depth_num_intervals = 4.  depths = [0.25, 0.4166..., 0.5833..., 0.75]
    else:
        logger.error(f'For {desc}, value_min {value_min} needs to be smaller than value_max {value_max}')
        raise ValueError
    
    if int_rounding:
        values = [int(i) for i in values]

    return values


def count_words(input_string):
    pattern = r"[^\s]+[\s]*"
    words = re.findall(pattern, input_string)
    return len(words)

def truncate_string_by_word_count(input_string, word_count_end, word_count_start = 0):
    # Regex to match words including punctuation and newlines
    if word_count_start < word_count_end:
        pattern = r"[^\s]+[\s]*"
        words = re.findall(pattern, input_string)
        truncated_words = words[word_count_start : word_count_end]
        return ''.join(truncated_words).strip()
    elif word_count_end == word_count_start:
        return ''
    else:
        logger.error(f'Potential error: word_count_start = {word_count_start}; word_count_end = {word_count_end}')
        return ''

def make_raw_answer(retrieval_target_len, retrieval_target_style, retrieval_target_wrapper, exclusions = []):

    char_candidates = 'abcdefghijklmnopqrstuvwxyz'
    digit_candidates = '0123456789'
    city_candidates = ['Chicago', 'Yangon', 'Antananarivo', 'Colombo', 'Almaty', 'Sydney', 'Chicago', 'Mexico City',
    'Seattle', 'Lagos', 'Amsterdam', 'Belgrade', 'Cairo', 'Baghdad', 'Damascus', 'Kigali', 'Dakar',
    'Dakar', 'Sofia', 'Kigali', 'Victoria', 'Tashkent', 'Mumbai', 'Barcelona', 'Almaty', 'Amman',
    'Toronto', 'Bratislava', 'Johannesburg', 'Thimphu', 'Bangkok', 'Santiago', 'Cairo', 'San Francisco',
    'Lagos', 'Amsterdam', 'Paris', 'Rabat', 'Santiago', 'Copenhagen', 'Madrid', 'Kigali',
    'Ho Chi Minh City', 'Sarajevo', 'Delhi', 'Istanbul', 'Ho Chi Minh City', 'Khartoum', 'Helsinki',
    'Doha', 'Istanbul', 'Kuala Lumpur', 'Budapest', 'Shanghai', 'Moscow', 'Los Angeles', 'Oslo',
    'Johannesburg', 'Berlin', 'Bangalore', 'Tokyo', 'Melbourne', 'Barcelona', 'Chicago', 'Port Louis',
    'Lisbon', 'Nairobi', 'Kampala', 'Lima', 'Maputo', 'Vancouver', 'Dubai', 'Khartoum', 'Jakarta',
    'Madrid', 'Yerevan', 'Beirut', 'Athens', 'Chicago', 'Paris', 'Bucharest', 'Copenhagen', 'Brussels',
    'Damascus', 'Seattle', 'Los Angeles', 'Yerevan', 'Victoria', 'Tunis', 'Astana', 'Seoul',
    'Buenos Aires', 'Bangkok', 'Colombo', 'Brussels', 'Khartoum', 'Doha', 'San Francisco', 'Vienna', 'Jakarta']

    if retrieval_target_style == 'char':
        target_candidates = char_candidates
    elif retrieval_target_style == 'int':
        target_candidates = digit_candidates
    elif retrieval_target_style == 'int_n_char':
        target_candidates = char_candidates + digit_candidates
    elif retrieval_target_style == 'city':
        target_candidates = city_candidates
    else:
        logger.error(f'Invalid retrieval_target_style input: {retrieval_target_style}.')
        raise ValueError

    while True:
        if not retrieval_target_style == 'city':
            raw_answer = str(''.join(random.choice(target_candidates) for _ in range(retrieval_target_len)))
            if raw_answer not in exclusions:
                break
        else:
            raw_answer = random.choice(target_candidates)
            if raw_answer not in exclusions:
                break

    if retrieval_target_wrapper == 'naked':
        wrapped_answer = raw_answer
    elif retrieval_target_wrapper == "double_quotes":
        wrapped_answer = "\"" + raw_answer + "\""
    elif retrieval_target_wrapper == "code":
        wrapped_answer = "`" + raw_answer + "`"
    else:
        logger.error(f'Invalid retrieval_target_wrapper input: {retrieval_target_wrapper}.')
        raise ValueError
    
    return raw_answer, wrapped_answer

def make_passkey_retrieval_target(retrieval_target_len, retrieval_target_style, retrieval_target_template, retrieval_target_placeholder, retrieval_target_wrapper):

    raw_answer, wrapped_answer = make_raw_answer(retrieval_target_len=retrieval_target_len, retrieval_target_style=retrieval_target_style, retrieval_target_wrapper=retrieval_target_wrapper)
    retrieval_target = retrieval_target_template.replace(retrieval_target_placeholder, wrapped_answer)

    return retrieval_target, raw_answer



def make_magic_city_number_retrieval_targets(retrieval_target_len, retrieval_target_style, retrieval_target_template, city_placeholder, number_placeholder, retrieval_target_wrapper, distraction_num):

    if distraction_num < 0:
        logger.error(f'Invalid distraction_num input: {distraction_num}.')
        raise ValueError
    
    retrieval_targets = []
    wrapped_questions = []
    raw_questions = []
    raw_answers = []

    
    for _ in range(distraction_num + 1):
        raw_city_question, wrapped_city_question = make_raw_answer(retrieval_target_len=1, retrieval_target_style='city', retrieval_target_wrapper=retrieval_target_wrapper, exclusions=raw_questions)
        raw_number_answer, wrapped_number_answer = make_raw_answer(retrieval_target_len=retrieval_target_len, retrieval_target_style=retrieval_target_style, retrieval_target_wrapper=retrieval_target_wrapper, exclusions=raw_answers)
        retrieval_target = retrieval_target_template.replace(city_placeholder, wrapped_city_question)
        retrieval_target = retrieval_target.replace(number_placeholder, wrapped_number_answer)

        raw_questions.append(raw_city_question)
        wrapped_questions.append(wrapped_city_question)
        raw_answers.append(raw_number_answer)
        retrieval_targets.append(retrieval_target)


    wrapped_question = wrapped_questions.pop(0)
    raw_answer = raw_answers.pop(0)
    retrieval_target = retrieval_targets.pop(0)

    distraction_retrieval_targets = retrieval_targets
    distraction_raw_answers = raw_answers
    

    return retrieval_target, wrapped_question, raw_answer, distraction_retrieval_targets, distraction_raw_answers

def make_background(background_dir, background_len, background_filling_style):
    if os.path.isfile(background_dir):
        with open(background_dir, 'r') as background_text_f:
            background_text = background_text_f.read()
            background_text_len = count_words(background_text)
            background_text_LUT = [{"background_text": background_text, "background_text_len": background_text_len}]
    elif os.path.isdir(background_dir):
        file_dirs = [str(background_dir) + "/" + str(f) for f in os.listdir(background_dir) if os.path.isfile(os.path.join(background_dir, f))]
        background_text_LUT = []
        for a_file in file_dirs:
            with open(a_file, 'r') as background_text_f:
                background_text = background_text_f.read()
                background_text_len = count_words(background_text)
                background_text_LUT.append({"background_text": background_text, "background_text_len": background_text_len})
    else:
        logger.error(f'Invalid background_dir input: {background_dir}.')
        raise ValueError
    
    if background_filling_style == 'ordered_repetition':
        background_text_circular_list = cycle(background_text_LUT)
        current_background_len = 0
        background_text = ''
        
        for i in background_text_circular_list:
            i_background_text = i['background_text']
            i_background_text_len = i['background_text_len']

            background_text += i_background_text
            current_background_len += i_background_text_len
            if current_background_len >= background_len:
                break
        background_text = truncate_string_by_word_count(background_text, word_count_end = background_len)
    elif background_filling_style == 'random_repetition':
        raise NotImplementedError
    else:
        logger.error(f'Invalid background_filling_style input: {background_filling_style}.')
        raise ValueError


    return background_text
        

def make_content(depths, retrieval_targets, background_text):
    
    retrieval_targets = retrieval_targets + ['']
    
    # logger.info(f'depths: {depths}')
    # logger.info(f'retrieval_targets: {retrieval_targets}')
    

    background_len = count_words(background_text)

    retrieval_target_start_positions = [int(background_len * depth) for depth in depths] # [start_pos_1, start_pos_2, ...]

    retrieval_target_start_positions = [0] + retrieval_target_start_positions + [background_len] # [0, start_pos_1, start_pos_2, ..., background_len]
    
    retrieval_target_start_n_end_positions = [(retrieval_target_start_positions[i], retrieval_target_start_positions[i+1]) for i in range(len(retrieval_target_start_positions) - 1)]
    # Till here, retrieval_target_start_end_positions = [ (0, start_pos_1), (start_pos_1, start_pos_2), (start_pos_2, star_pos_3)]

    # logger.info(f'depths: {depths}; retrieval_targets: {retrieval_targets}')
    # logger.info(f'retrieval_target_start_n_end_positions: {retrieval_target_start_n_end_positions}')
    
    background_chunks = [truncate_string_by_word_count(background_text, word_count_start = i, word_count_end = j) for i, j in retrieval_target_start_n_end_positions]

    background_chunks_n_retrieval_targets = []

    for background_chunk, retrieval_target in zip(background_chunks, retrieval_targets):
        background_chunks_n_retrieval_targets.append(background_chunk)
        background_chunks_n_retrieval_targets.append(retrieval_target)
        

    content = ' '.join(background_chunks_n_retrieval_targets)

    
    return content


def make_full_input_for_all_iterations(background_len, depth, config):
    if config['eval_params']['dataset'] == 'passkey_retrieval': 
        return make_full_passkey_retrieval_input(background_len, depth, config)
    elif config['eval_params']['dataset'] == 'magic_city_number_retrieval': 
        return make_full_magic_city_number_retrieval_input(background_len, depth, config)
    else:
        logger.error(f"Invalid config['eval_params']['dataset'] input: {config['eval_params']['dataset']}.")
        raise ValueError


def make_full_passkey_retrieval_input(background_len, depth, config):
    
    # full_input = instructions + content
    # content = background + retrieval target
    # retrieval target = retrieval template + raw answer

    background_text = make_background(background_dir = config['eval_params']['background_dir'], background_len = background_len, background_filling_style = config['eval_params']['background_filling_style'])
    
    full_inputs = []
    contents = []
    raw_answers = []    
    instructions = []
    retrieval_questions = []

    for _ in range(config['eval_params']['depth_num_iterations']):
        retrieval_target, raw_answer = make_passkey_retrieval_target(retrieval_target_len = config['eval_params']['retrieval_target_len'], 
                                                            retrieval_target_style = config['eval_params']['retrieval_target_style'], 
                                                            retrieval_target_template = config['eval_params']['retrieval_target_template'], 
                                                            retrieval_target_placeholder = config['eval_params']['retrieval_target_placeholder'], 
                                                            retrieval_target_wrapper = config['eval_params']['retrieval_target_wrapper'])
        
        content = make_content(depths = [depth], retrieval_targets = [retrieval_target], background_text = background_text)
        retrieval_question = config['eval_params']['retrieval_question']
        instruction = config['eval_params']['instruction']
        if config['eval_params']['instruction_position'] == 'prefix':
            full_input = instruction + content + retrieval_question
        elif config['eval_params']['instruction_position'] == 'surfix':
            full_input = content + instruction + retrieval_question
        else:
            logger.error(f"Invalid config['eval_params']['instruction_position'] input: {config['eval_params']['instruction_position']}.")
            raise ValueError

        contents.append(content)
        raw_answers.append(raw_answer)
        full_inputs.append(full_input)
        instructions.append(instruction)
        retrieval_questions.append(retrieval_question)


    # len() of any of the three return subjects below is config['eval_params']['depth_num_iterations']
    return full_inputs, contents, raw_answers, instructions, retrieval_questions



def make_full_magic_city_number_retrieval_input(background_len, depth, config):
    
    # full_input = instructions + content
    # content = background + retrieval target
    # retrieval target = retrieval template + raw answer

    background_text = make_background(background_dir = config['eval_params']['background_dir'], background_len = background_len, background_filling_style = config['eval_params']['background_filling_style'])
    
    full_inputs = []
    contents = []
    raw_answers = [] 
    instructions = []
    retrieval_questions = [] 
    distraction_allowed_answers = [] # only useful when we have distraction_allowed metrics

    #retrieval_target_len, retrieval_target_style, retrieval_target_template, city_placeholder, number_placeholder, retrieval_target_wrapper, distraction_num):

    for _ in range(config['eval_params']['depth_num_iterations']):
        retrieval_target, wrapped_question, raw_answer, distraction_retrieval_targets, distraction_raw_answers = make_magic_city_number_retrieval_targets(retrieval_target_len = config['eval_params']['retrieval_target_len'], 
                                                            retrieval_target_style = config['eval_params']['retrieval_target_style'], 
                                                            retrieval_target_template = config['eval_params']['retrieval_target_template'], 
                                                            city_placeholder = config['eval_params']['city_placeholder'],
                                                            number_placeholder = config['eval_params']['number_placeholder'],
                                                            retrieval_target_wrapper = config['eval_params']['retrieval_target_wrapper'],
                                                            distraction_num = config['eval_params']['distraction_num'])
        
        depths, retrieval_targets = get_magic_city_number_depths_n_targets(retrieval_target = retrieval_target, retrieval_target_depth = depth, distraction_retrieval_targets = distraction_retrieval_targets)
        
        content = make_content(depths = depths, retrieval_targets = retrieval_targets, background_text = background_text)


        retrieval_question = config['eval_params']['retrieval_question']
        retrieval_question = retrieval_question.replace(config['eval_params']['city_placeholder'], wrapped_question)
        instruction = config['eval_params']['instruction']
        if config['eval_params']['instruction_position'] == 'prefix':
            full_input = instruction + content + retrieval_question
        elif config['eval_params']['instruction_position'] == 'surfix':
            full_input = content + instruction + retrieval_question
        else:
            logger.error(f"Invalid config['eval_params']['instruction_position'] input: {config['eval_params']['instruction_position']}.")
            raise ValueError

        contents.append(content)
        raw_answers.append(raw_answer)
        full_inputs.append(full_input)
        instructions.append(instruction)
        retrieval_questions.append(retrieval_question)
        distraction_allowed_answers.append([raw_answer] + distraction_raw_answers)


    # len() of any of the three return subjects below is config['eval_params']['depth_num_iterations']
    return full_inputs, contents, raw_answers, instructions, retrieval_questions, distraction_allowed_answers


def get_magic_city_number_depths_n_targets(retrieval_target, retrieval_target_depth, distraction_retrieval_targets):


    n_distraction_targets = len(distraction_retrieval_targets)

    n_distraction_targets_before_retrieval_target = int(n_distraction_targets * retrieval_target_depth)
    n_distraction_targets_after_retrieval_target = n_distraction_targets - n_distraction_targets_before_retrieval_target

    depths = [i * retrieval_target_depth/(n_distraction_targets_before_retrieval_target + 1) for i in range(1, n_distraction_targets_before_retrieval_target + 1)]
    depths.append(retrieval_target_depth)
    depths = depths + [retrieval_target_depth + i * (1 - retrieval_target_depth)/(n_distraction_targets_after_retrieval_target + 1) for i in range(1, n_distraction_targets_after_retrieval_target + 1)]
    # Equally divide the depths before and after the retrieval_target_depth in proportion to the value of retrieval_target_depth.

    # Assume retrieval_target_depth = 0.4, len(distraction_retrieval_targets) = 10, we'd have n_distraction_targets_before_retrieval_target = 4 and n_distraction_targets_after_retrieval_target = 6.
    # So 4 distractions equally inserted before retrieval_target_depth, and 6 distractions after. 
    # Which means we'd have depths = [0.08, 0.16, 0.24, 0.32, 0.4, 0.486, 0.571, 0.657, 0.743, 0.829, 0.914]. 

    retrieval_targets = []
    for depth in depths:
        if depth != retrieval_target_depth:
            retrieval_targets.append(distraction_retrieval_targets.pop())
        else:
            retrieval_targets.append(retrieval_target)


    return depths, retrieval_targets


def process_raw_exp_results(raw_exp_results, metrics):
    # raw_exp_results = [ {background_len: , depth: , iteration: , answer: , content: , full_input: , response: }, {}, ...]    

    processed_results = {}
    for i in raw_exp_results:
        if i['background_len'] not in processed_results:
            processed_results[i['background_len']] = {}
        if i['depth'] not in processed_results[i['background_len']]:
            if "distraction_allowed_exact_match" in metrics or "distraction_allowed_partial_match" in metrics:
                processed_results[i['background_len']][i['depth']] = {"answers": [], "responses": [], "distraction_allowed_answers": []}
            else: 
                processed_results[i['background_len']][i['depth']] = {"answers": [], "responses": []}

        processed_results[i['background_len']][i['depth']]['answers'].append(i['answer'])
        processed_results[i['background_len']][i['depth']]['responses'].append(i['response'])
        if "distraction_allowed_exact_match" in metrics or "distraction_allowed_partial_match" in metrics:
            processed_results[i['background_len']][i['depth']]['distraction_allowed_answers'].append(i['distraction_allowed_answers'])
            

    # raw_results = { background_len_1: {depth_1: {answers: [answer_of_iteration_1, ...], responses: [response_of_iteration_1, ...]}, depth_2: {}, ...}, background_len_2: {}, ...}
    raw_results = copy.deepcopy(processed_results)

    for background_len in processed_results.keys():
        for depth, depth_v in processed_results[background_len].items():
            answers = depth_v['answers']
            responses = depth_v['responses']
            
            processed_results[background_len][depth] = {}
            for metric in metrics:
                if metric == "distraction_allowed_exact_match" or metric == "distraction_allowed_partial_match":
                    result_by_metric = sum(eval_metrics.eval_by_metric(depth_v['distraction_allowed_answers'], responses, metric))/len(responses)
                    processed_results[background_len][depth][metric] = result_by_metric
                else:
                    result_by_metric = sum(eval_metrics.eval_by_metric(answers, responses, metric))/len(responses)
                    processed_results[background_len][depth][metric] = result_by_metric

    # Till here, processed_results = { background_len_1: {depth_1: {metric_1: , metric_2: , ...}, depth_2: {}, ...}, background_len_2: {}, ...}
    
    overall_results = {}
    background_len_wise_results = {}
    for metric in metrics:
        all_reported_metric = []
        for background_len_k, background_len_v in processed_results.items():
            # background_len_v = {0: {'exact_match': 0.0, 'partial_match': 0.1}, 1: {'exact_match': 0.0, 'partial_match': 0.8}}
            all_reported_metric_under_certain_background_len = []
            for depth_v in background_len_v.keys():
                all_reported_metric.append(background_len_v[depth_v][metric])
                all_reported_metric_under_certain_background_len.append(background_len_v[depth_v][metric])
            
            if background_len_k not in background_len_wise_results:
                background_len_wise_results[background_len_k] = {}

            background_len_wise_results[background_len_k][metric] = sum(all_reported_metric_under_certain_background_len)/len(all_reported_metric_under_certain_background_len)
            
        overall_results[metric] = sum(all_reported_metric)/len(all_reported_metric)


    processed_results['background_len_wise_results'] = background_len_wise_results
    processed_results['overall_results'] = overall_results

    # Till here, processed_results = { background_len_1: {depth_1: {metric_1: , metric_2: , ...}, depth_2: {}, ...}, background_len_2: {}, ..., overall_results: {metric_1: , metric_2: , ...}}
    return processed_results, raw_results


def check_if_out_of_context_window(longest_input, model_max_len, tokenizer, out_of_max_len_allowed):
    
    longest_input_token_len = len(tokenizer.tokenize(longest_input))

    if longest_input_token_len > model_max_len:
        if not out_of_max_len_allowed:
            logger.error(f'Longest input ({longest_input_token_len} tokens) exceeds model_max_len ({model_max_len} tokens) when out_of_max_len_allowed is {out_of_max_len_allowed}.')
            raise ValueError
        else:
            logger.error(f'Longest input ({longest_input_token_len} tokens) exceeds model_max_len ({model_max_len} tokens), but out_of_max_len_allowed is {out_of_max_len_allowed} so experiment continuous.')

    else: 
        logger.info(f'Longest input ({longest_input_token_len} tokens) is below model_max_len ({model_max_len} tokens).')

    

