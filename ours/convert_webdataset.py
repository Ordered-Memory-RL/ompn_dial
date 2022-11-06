import webdataset as wds
from utils import DictList
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

DATASET = "jacopinpad_4"
TRAIN_PROPORTION = 0.8


def padded(ndarray):
    lis = []
    for i in ndarray:
        lis.append(torch.tensor(i))
    # return pad_sequence(lis, batch_first=True)
    return lis


f = open(f"dataset/{DATASET}.pkl", "rb")
d = pickle.load(f)
keep = ["states", "actions", "gt_onsets", "tasks"]
dk = {k: v for k, v in d.items() if k in keep}
from utils import point_of_change

poc = []
for g in dk["gt_onsets"]:
    poc.append(point_of_change(g))

poc = np.array(poc)
dk = {k: np.array(v) for k, v in dk.items()}
dk["points_of_change"] = poc
dk["gt_onsets"] = padded(dk["gt_onsets"])
dk["states"] = padded(dk["states"])
dk["actions"] = padded(dk["actions"])

dl = DictList(dk)
sink = wds.TarWriter("dataset/jacopinpad_4_train.tar")
for i in range(int(1400 * TRAIN_PROPORTION)):
    o = dict(dl[i])
    o = {f"{k}.pyd": v for k, v in o.items()}
    o["__key__"] = str(i)
    sink.write(o)
sink.close()


sink = wds.TarWriter("dataset/jacopinpad_4_val.tar")
for i in range(int(1400 * (1 - TRAIN_PROPORTION))):
    o = dict(dl[i])
    o = {f"{k}.pyd": v for k, v in o.items()}
    o["__key__"] = str(i)
    sink.write(o)
sink.close()
