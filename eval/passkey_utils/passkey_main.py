import logging

logger = logging.getLogger("main")

import eval.passkey_utils.passkey_utils as passkey_utils
import eval.passkey_utils.eval_metrics as eval_metrics


def prepare_passkey_retrieval_input(config):
    eval_params = config['eval_params']
    # pipeline_params = config['pipeline_params']

    background_len_min = eval_params['background_len_min']
    background_len_max = eval_params['background_len_max']
    n_background_lens = eval_params['n_background_lens']
    depth_min = eval_params['depth_min']
    depth_max = eval_params['depth_max']
    n_depths = eval_params['n_depths']
    depth_num_iterations = eval_params['depth_num_iterations']

    background_lens = passkey_utils.get_intermediate_values_within_min_max(background_len_min, background_len_max,
                                                                           n_background_lens, desc='background_lens',
                                                                           int_rounding=True)
    depths = passkey_utils.get_intermediate_values_within_min_max(depth_min, depth_max, n_depths, desc='depths')

    logger.info(f'Evaluate on background_lens: {background_lens}')
    logger.info(f'Evaluate on depths: {depths}')

    # raw_exp_results = [ {background_len: , depth: , iteration: , answer: , content: , retrieval question: , full_input: }, {}, ...]
    raw_exp_results = []
    for background_len in background_lens:
        for depth in depths:
            if config['eval_params']['dataset'] == 'magic_city_number_retrieval':
                # if the task is magic city number, then raw_exp_results will also contain a key "distraction_allowed_answers": [" ", " ",...]
                full_inputs, contents, answers, instructions, retrieval_questions, distraction_allowed_answers = passkey_utils.make_full_input_for_all_iterations(
                    background_len, depth, config)
                for i in range(depth_num_iterations):
                    raw_exp_results.append(
                        {"background_len": background_len, "depth": depth, "iteration": i, "answer": answers[i],
                        "content": contents[i], "full_input": full_inputs[i], "instruction": instructions[i],
                        "retrieval question": retrieval_questions[i], "distraction_allowed_answers": distraction_allowed_answers[i]})

            else:
                full_inputs, contents, answers, instructions, retrieval_questions = passkey_utils.make_full_input_for_all_iterations(
                    background_len, depth, config)

                for i in range(depth_num_iterations):
                    raw_exp_results.append(
                        {"background_len": background_len, "depth": depth, "iteration": i, "answer": answers[i],
                        "content": contents[i], "full_input": full_inputs[i], "instruction": instructions[i],
                        "retrieval question": retrieval_questions[i]})

    logger.info(
        f'All experiment {len(raw_exp_results)} inputs generated ({len(background_lens)} background_lens; {len(depths)} depths; {depth_num_iterations} iterations for each depth level)')
    # logger.info(f"RAW EXP RESULTS is {raw_exp_results}")\

    return raw_exp_results
