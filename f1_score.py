# f1_score.py
from collections import Counter

def f1_score(prediction, ground_truth):
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()

    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common_tokens.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def average_f1_score(predictions, ground_truths):
    total_f1_score = sum(f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths))
    return total_f1_score / len(predictions)
