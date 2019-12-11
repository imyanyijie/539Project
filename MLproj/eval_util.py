import numpy as np
import torch

def acc_topk(preds, labels, top_k=1):
    tfs = []
    for i in range(top_k):
        tfs.append((preds[:, i] == labels))

    acc = np.sum((np.sum(tfs, axis=0) > 0)) / preds.shape[0]

    return acc


def res_topk(outputs, dir_num_to_name, top_k=1):
    _, ind = torch.topk(outputs, top_k)

    res = [dir_num_to_name[str(ind[0, i].item())] for i in range(top_k)]

    return res