import collections
import logging
import re
import string
from difflib import SequenceMatcher

logger = logging.getLogger("main")

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', '', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def eval_by_metric(answers, responses, metric):
    if len(answers) != len(responses):
        logger.error(
            f"Invalid input: len(answers) ({len(answers)}) must be equal to len(responses) ({len(responses)}).")
        raise ValueError

    if metric == 'exact_match':
        acc_by_metric = exact_match(answers, responses)
    elif metric == 'partial_match':
        acc_by_metric = partial_match(answers, responses)
    elif metric == 'F1':
        acc_by_metric = F1_score(answers, responses)
    elif metric == 'distraction_allowed_exact_match':
        # here the answers should be a list of list of answers
        # answers = [[" "," ",...], [" ", " ",...], ...]
        acc_by_metric = exact_match(answers, responses)
    elif metric == 'distraction_allowed_partial_match':
        # here the answers should be a list of list of answers
        # answers = [[" "," ",...], [" ", " ",...], ...]
        acc_by_metric = distraction_allowed_partial_match(answers, responses)
    else:
        logger.error(f"Invalid metric input: {metric}.")
        raise ValueError

    return acc_by_metric


def distraction_allowed_partial_match(list_of_answers, responses):
    result = []
    for answers, response in zip(list_of_answers, responses):
        max_pm = -float("inf")
        for answer in answers:
            pm_score = partial_match([answer], [response])
            pm_score = pm_score[0]
            if pm_score > max_pm:
                max_pm = pm_score
        result.append(max_pm)
    return result


def exact_match(answers, responses):
    result = []
    for answer, response in zip(answers, responses):
        response = normalize_answer(response)
        if isinstance(answer, str):
            if normalize_answer(answer) in response:
                result.append(1)
        elif isinstance(answer, list):
            if len(answer) == 0:
                result.append(0)
            elif not isinstance(answer[0], list):
                result.append(int(any([normalize_answer(a) in response for a in answer])))
            else:
                temp = 0
                for ans in answer:
                    temp += int(any([normalize_answer(a) in response for a in ans]))
                result.append(temp/len(answer))
    return result


def partial_match(answers, responses):
    def find_longest_common_substring(answer, response):
        sequence_matcher = SequenceMatcher(None, answer, response, autojunk=False)
        match = sequence_matcher.find_longest_match(0, len(answer), 0, len(response))
        return match.size / len(answer)

    result = []
    for answer, response in zip(answers, responses):
        result.append(find_longest_common_substring(answer, response))

    return result


def F1_score(answers, responses):
    def compute_f1(a_gold: str, a_pred: str):
        def get_tokens(s: str):
            if not s: return []
            return s.split()

        gold_toks = get_tokens(a_gold)
        pred_toks = get_tokens(a_pred)
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(gold_toks == pred_toks)
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_toks)
        recall = 1.0 * num_same / len(gold_toks)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    result = []
    for answer, response in zip(answers, responses):
        if isinstance(answer, str):
            if answer in response:
                result.append(compute_f1(answer, response))
        elif isinstance(answer, list):
            result.append(max(compute_f1(a, response) for a in answer))
    return result
