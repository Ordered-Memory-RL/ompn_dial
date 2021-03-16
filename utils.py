import torch
from absl import logging
from torch import nn as nn
import numpy as np


def sequence_mask(lengths, max_len=None):
    """ True for i < lengths. False for i >= lengths """
    max_len = lengths.max().item() if max_len is None else max_len
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def point_of_change(onsets):
    """ Return: an list of change points. Such that change[i] is the ending of ith segment.
    change[-1] is the last position """
    prev_ele = onsets[0]
    result = []
    for idx in range(1, len(onsets)):
        curr_ele = onsets[idx]
        if curr_ele != prev_ele:
            result.append(idx - 1)
        prev_ele = curr_ele
    result.append(len(onsets) - 1)
    return result


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.
    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
        >>> d.c = [[7], [8]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5], "c": [7]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def __repr__(self):
        _d = {k: dict.__getitem__(self, k) for k in self.keys()}
        return _d.__repr__()

    def append(self, other):
        """ Append another DictList with one element """
        if not isinstance(other, DictList):
            other = DictList(other)
        for key in other:
            if key not in self:
                self.__setattr__(key, [])
            existing = self.__getattr__(key)
            existing.append(other.__getattr__(key))

    def __add__(self, other):
        """ Concat """
        if not isinstance(other, DictList):
            other = DictList(other)
        res = {}
        for key in other:
            if key not in self:
                self.__setattr__(key, [])
            existing = self.__getattr__(key)
            res[key] = existing + other.__getattr__(key)
        for key in self:
            if key not in res:
                res[key] = self.__getattr__(key)
        return DictList(res)

    def apply(self, func):
        """ Apply the same func to each element """
        for key, val in self.items():
            self.__setattr__(key, func(val))

    def report(self):
        for key, val in self.items():
            print(key, val.shape)


class Metric:
    """ A general metric object for recording mean """
    def __init__(self, fields):
        self.data = {name: 0 for name in fields}
        self.count = {name: 0 for name in fields}

    def mean(self):
        return {name: self.data[name] / self.count[name] if self.count[name] != 0 else 0
                for name in self.data}

    def reset(self):
        self.data = {name: 0 for name in self.data}
        self.count = {name: 0 for name in self.data}

    def accumulate(self, stats, counts):
        if isinstance(stats, DictList):
            stats = {k: v for k, v in stats.items()}
        for name in stats:
            if self.data.get(name, None) is None:
                self.data[name] = 0
                self.count[name] = 0

            if isinstance(stats[name], torch.Tensor):
                self.data[name] += stats[name].item() * counts
            elif isinstance(stats[name], float):
                self.data[name] += stats[name] * counts
            self.count[name] += counts

    def __str__(self):
        result = self.mean()
        return "\t".join(['{}:{:.4f}'.format(key, val) for key, val in result.items()])


class SketchEmbedding(nn.Module):
    """ An env emb from hard coding """
    def __init__(self, emb_size):
        super(SketchEmbedding, self).__init__()
        self.sketch_embedding = nn.Embedding(num_embeddings=10,
                                             embedding_dim=emb_size)
        self.emb_size = emb_size

    def forward(self, sketchs, sketch_lengths):
        sketchs_emb = self.sketch_embedding(sketchs)
        sketch_mask = sequence_mask(sketch_lengths).float()
        return (sketchs_emb * sketch_mask.unsqueeze(-1)).sum(-2)


class GRUSketchEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(GRUSketchEmbedding, self).__init__()
        self.sketch_embedding = nn.Embedding(num_embeddings=10,
                                             embedding_dim=emb_size)
        self.emb_size = emb_size
        self.gru = torch.nn.GRU(input_size=emb_size, hidden_size=emb_size, batch_first=True)

    def forward(self, sketchs, sketch_lengths):
        sketchs = sketchs[:, :sketch_lengths.max().item()]
        sketchs_emb = self.gru(self.sketch_embedding(sketchs))[0]
        batch_id = torch.arange(sketchs_emb.shape[0], device=sketchs_emb.device)
        final_states = sketchs_emb[batch_id, sketch_lengths - 1]
        return final_states


class NoEnvEmbedding(nn.Module):
    def __init__(self, emb_size):
        super(NoEnvEmbedding, self).__init__()
        self.emb_size = emb_size
        self.zeros = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, sketchs, sketch_lengths):
        final_shape = list(sketchs.shape[:-1]) + [self.emb_size]
        return self.zeros.repeat(*final_shape)


def get_env_encoder(env_arch, emb_size):
    if env_arch == 'sketch':
        return SketchEmbedding(emb_size)
    elif env_arch == 'grusketch':
        return GRUSketchEmbedding(emb_size)
    elif env_arch == 'noenv':
        return NoEnvEmbedding(emb_size)
    else:
        raise ValueError


def get_action_dist(mean):
    return torch.distributions.MultivariateNormal(loc=mean,
                                                  covariance_matrix=torch.eye(mean.shape[-1],
                                                                              device=mean.device) * 0.6)


class Normaliser(nn.Module):
    def __init__(self, mean, std):
        super(Normaliser, self).__init__()
        self.mean = nn.Parameter(torch.tensor(mean).float(), requires_grad=False)
        self.std = nn.Parameter(torch.tensor(std).float(), requires_grad=False)

    def normalize(self, orig_value):
        return (orig_value - self.mean) / self.std

    def recover(self, normalized):
        return normalized * self.std + self.mean

def logging_metrics(nb_frames, steps, metrics, writer, prefix):
    for env_name, metric in metrics.items():
        line = ['[{}][{}] steps={}'.format(prefix, env_name, steps)]
        for k, v in metric.items():
            line.append('{}: {:.4f}'.format(k, v))
        logging.info('\t'.join(line))
    mean_val_metric = DictList()
    for metric in metrics.values():
        mean_val_metric.append(metric)
    mean_val_metric.apply(lambda t: torch.mean(torch.tensor(t)))
    for k, v in mean_val_metric.items():
        writer.add_scalar(prefix + '/' + k, v.item(), nb_frames)
    writer.flush()