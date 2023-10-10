
import numpy as np

def average(preds_list):
    return np.mean(preds_list, axis=0)

def majority_vote(preds_list):
    stacked = np.stack(preds_list, axis=1)
    return (np.sum(stacked, axis=1) > len(preds_list) / 2).astype(int)

