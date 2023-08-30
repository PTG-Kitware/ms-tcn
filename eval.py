#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py
import os
import argparse

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


plt.rcParams.update({"font.size": 35})


def read_file(path):
    with open(path, "r") as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == y_label[x] for x in range(len(y_label))]
        )
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def eval(file_list: str, ground_truth_path: str,
        recog_path: str, eval_output: str):
    """Run evaluation

    :param file_list: Path to the test bundle file
    :param ground_truth_path: Path to the ``groundTruth`` folder
    :param recog_path: Path to the predicition results
        (obtained by running with ``action=predict``)
    :param eval_output: Path to save the metrics and figures to
    """
    list_of_videos = read_file(file_list).split("\n")[:-1]

    overlap = [0.1, 0.25, 0.5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    true_idxs = []
    pred_idxs = []
    for vid in list_of_videos:
        gt_file = os.path.join(ground_truth_path, vid)
        gt_content = read_file(gt_file).split("\n")[0:-1]

        recog_file = os.path.join(recog_path, vid)
        recog_content = read_file(recog_file).split("\n")[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

            true_idxs.append(gt_content[i])
            pred_idxs.append(recog_content[i])

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    # Metrics
    with open(f"{eval_output}/metrics.txt", "w") as f:
        acc = "Acc: %.4f" % (100 * float(correct) / total)
        print(acc)
        f.write(f"{acc}\n")

        edit = "Edit: %.4f" % ((1.0 * edit) / len(list_of_videos))
        print(edit)
        f.write(f"{edit}\n")
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100
            f1_str = "F1@%0.2f: %.4f" % (overlap[s], f1)
            print(f1_str)
            f.write(f"{f1_str}\n")
    
    # Create confusion matrix
    cm = confusion_matrix(
        true_idxs, pred_idxs,
        normalize="true"
    )

    fig, ax = plt.subplots(figsize=(100, 100))
    sn.heatmap(cm, annot=True, fmt=".2f", ax=ax)
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Label",
        ylabel="True Label",
    )
    fig.savefig(f"{eval_output}/confusion_mat.png", pad_inches=5)
    plt.close(fig)

    recall = np.diag(cm) / np.sum(cm, axis=1)  # rows
    print(f"cm recall: {np.mean(recall)}")
    precision = np.diag(cm) / np.sum(cm, axis=0)  # columns
    print(f"cm precision: {np.mean(precision)}")

    print(f"Saved eval results to: {eval_output}")
