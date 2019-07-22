import numpy as np
from sklearn import metrics


def prequential_measure(pred, label, reset_pos=np.array([0])):
    n = pred.size
    pq_result = {}

    pq_result['gm'] = np.zeros(n)
    pq_result['f1'] = np.zeros(n)
    pq_result['auc'] = np.zeros(n)
    pq_result['rec'] = np.zeros(n)

    for i in range(n):
        start_pos = sum(i >= reset_pos) - 1
        pq_result['gm'][i] = gm_measure(pred[reset_pos[start_pos]:i + 1], label[reset_pos[start_pos]:i + 1])
        pq_result['f1'][i] = f1_measure(pred[reset_pos[start_pos]:i + 1], label[reset_pos[start_pos]:i + 1])
        pq_result['auc'][i] = auc_measure(pred[reset_pos[start_pos]:i + 1], label[reset_pos[start_pos]:i + 1])
        pq_result['rec'][i] = rec_measure(pred[reset_pos[start_pos]:i + 1], label[reset_pos[start_pos]:i + 1])

    return pq_result


def gm_measure(pred, label):
    label = label.reshape(-1)
    tp = sum(np.bitwise_and(label == 1, pred == 1))
    fn = sum(np.bitwise_and(label == 1, pred == 0))
    tn = sum(np.bitwise_and(label == 0, pred == 0))
    fp = sum(np.bitwise_and(label == 0, pred == 1))

    if tp + fn == 0 or tn + fp == 0:
        gm = 0
    else:
        gm = np.sqrt(tp / (tp + fn) * tn / (tn + fp))

    return gm


def f1_measure(pred, label):
    label = label.reshape(-1)
    tp = sum(np.bitwise_and(label == 1, pred == 1))
    fn = sum(np.bitwise_and(label == 1, pred == 0))
    tn = sum(np.bitwise_and(label == 0, pred == 0))
    fp = sum(np.bitwise_and(label == 0, pred == 1))

    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1


def auc_measure(pred, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=1)
    try:
        auc = metrics.auc(fpr, tpr)
    except ValueError:
        auc = 0

    if np.isnan(auc):
        auc = 0

    return auc


def rec_measure(pred, label):
    if sum(label == 1) > sum(label == 0):
        min_class = 0
    else:
        min_class = 1

    label = label.reshape(-1)
    tp = sum(np.bitwise_and(label == min_class, pred == min_class))
    fn = sum(np.bitwise_and(label == min_class, pred == -min_class))

    if tp + fn == 0:
        rec = 0
    else:
        rec = tp / (tp + fn)

    return rec
