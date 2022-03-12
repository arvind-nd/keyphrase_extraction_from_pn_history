import os
import random
import torch
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed=args.seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def confusion_metrics(ground_truth, prediction):
    ground_truth_elements = set(ground_truth)
    prediction_elements = set(prediction)

    TP = len(ground_truth_elements & prediction_elements)
    FN = len(ground_truth_elements - prediction_elements)
    FP = len(prediction_elements - ground_truth_elements)

    return TP, FN, FP

def micro_average_f1(TP, FN, FP):
    tp = sum(TP)
    fn = sum(FN)
    fp = sum(FP)
    f1 = tp / (tp + 0.5*(fp + fn))
    return f1
