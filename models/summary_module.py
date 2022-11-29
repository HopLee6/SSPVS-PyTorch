import numpy as np
import math
from utils.knapsack import knapsack_ortools
import torch
from scipy import stats

# Ranking correlation metrics
f_kendalltau = lambda x, y: stats.kendalltau(stats.rankdata(-x), stats.rankdata(-y))[0]
f_spearmanr = lambda x, y: stats.spearmanr(stats.rankdata(-x), stats.rankdata(-y))[0]


def generate_summary(inputs, proportion=0.15, method="knapsack"):
    cps, n_frames, nfps = (
        inputs["change_points"],
        inputs["n_frames"],
        inputs["n_frame_per_seg"],
    )

    cps = cps[0].cpu().numpy()
    n_frames = int(n_frames.cpu())
    nfps = [int(p.cpu()) for p in nfps]

    n_segs = cps.shape[0]

    frame_scores = inputs["machine_score"]
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx, 0]), int(cps[seg_idx, 1] + 1)
        scores = frame_scores[start:end]
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))

    if method == "knapsack":
        # picks = knapsack_dp(seg_score, nfps, n_segs, limits)
        picks = knapsack_ortools(seg_score, nfps, n_segs, limits)
    elif method == "rank":
        order = np.argsort(seg_score)[::-1].tolist()
        picks = []
        total_len = 0
        for i in order:
            if total_len + nfps[i] < limits:
                picks.append(i)
                total_len += nfps[i]
    else:
        raise KeyError("Unknown method {}".format(method))

    summary = np.zeros((1), dtype=np.float32)  # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0)  # delete the first element
    return {"machine_summary": summary}


def evaluate_summary(inputs):
    machine_summary, user_summary = inputs["machine_summary"], inputs["user_summary"]
    eval_metric = inputs["eval_metric"][0]
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary[0].cpu().numpy().astype(np.float32)

    n_users, n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])

    f_scores = []
    prec_arr = []
    rec_arr = []

    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx, :]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.0
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)
    if eval_metric == "avg":
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == "max":
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]

    return {"f_score": final_f_score, "precison": final_prec, "recall": final_rec}


def evaluate_scores(inputs):
    machine_scores, user_scores = inputs["machine_score"], inputs["user_scores"]
    user_scores = user_scores[0].cpu().numpy().astype(np.float32)
    n_users, _ = user_scores.shape

    # Compute correlation with each annotator

    corrs_kendalltau = [
        f_kendalltau(machine_scores, user_scores[i]) for i in range(n_users)
    ]
    corrs_spearmanr = [
        f_spearmanr(machine_scores, user_scores[i]) for i in range(n_users)
    ]

    # Mean over all annotators
    corrs_kendalltau = np.mean(corrs_kendalltau)
    corrs_spearmanr = np.mean(corrs_spearmanr)

    return {"kendalltau": corrs_kendalltau, "spearmanr": corrs_spearmanr}


def upsample(scores, n_frames, positions):
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i == len(scores):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = scores[i]
    return frame_scores


def postprocess(p, boundary):
    if len(p) == 0:
        return
    temp = [
        p[i : i + 1].expand(boundary[i, 1] - boundary[i, 0] + 1)
        for i in range(p.shape[0])
    ]
    probs = torch.cat(temp, 0)
    return probs


def generate_machine_score(inputs):
    n_frames, positions, gtscore, cps = (
        inputs["n_frames"],
        inputs["picks"],
        inputs["gtscore"],
        inputs["change_points"],
    )
    cps = cps[0].cpu().numpy()
    n_frames = int(n_frames.cpu())
    positions = positions[0].cpu().numpy()
    gtscore = gtscore[0].cpu().numpy()
    gtscore_up = upsample(gtscore, n_frames, positions)

    ypred = inputs["frame_score"].squeeze()
    ypred = ypred.cpu().numpy()
    ypred_up = upsample(ypred, n_frames, positions)
    return {"machine_score": ypred_up, "gtscore_up": gtscore_up}


def summary_tool(inputs):
    inputs.update(generate_machine_score(inputs))
    inputs.update(evaluate_scores(inputs))
    inputs.update(generate_summary(inputs))
    inputs.update(evaluate_summary(inputs))
    return inputs
