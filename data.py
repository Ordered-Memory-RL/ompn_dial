"""
TACO dataloader
"""
from settings import DATASET_DIR
from utils import DictList
import os
from torch.nn.utils.rnn import pad_sequence
import pickle
from absl import logging
import random
import torch
import numpy as np
import time

__all__ = ['Dataloader']

MAX_TRAIN_LENGTH = 500


class RunningMeanMax:
    def __init__(self, dimension):
        self.val = np.zeros(dimension)
        self.count = 0
        self.max = np.zeros(dimension)

    def accumulate(self, val):
        self.val += val
        self.count += 1
        self.max = np.maximum(self.max, np.abs(val))

    @property
    def mean(self):
        return self.val / self.count


class Dataloader:
    def __init__(self, sketch_lengths, val_ratio):
        data = {}
        keys = ['states', 'actions', 'gt_onsets', 'tasks']
        for sketch_len in sketch_lengths:
            pkl_name = os.path.join(DATASET_DIR, 'jacopinpad_{}.pkl'.format(sketch_len))
            with open(pkl_name, 'rb') as f:
                data[sketch_len] = pickle.load(f)

        # Turn it into DictList
        for env_name in data:
            new_data = []
            trajs = data[env_name]
            trajs = [{k: trajs[k][i] for k in keys} for i in range(len(trajs['states']))]
            for traj in trajs:
                new_traj = DictList(traj)
                if len(new_traj) > MAX_TRAIN_LENGTH:
                    continue
                new_traj.done = [False]*(len(new_traj) - 1) + [True]
                new_data.append(new_traj)
            data[env_name] = new_data

        self.data = {'train': {}, 'val': {}}
        for env_name in data:
            _data = data[env_name]
            nb_data = len(_data)
            nb_val = int(val_ratio * nb_data)
            random.shuffle(_data)
            self.data['val'][env_name] = _data[:nb_val]
            self.data['train'][env_name] = _data[nb_val:]
            logging.info('{}: Train: {} Val: {}'.format(env_name, len(self.data['train'][env_name]),
                                                        len(self.data['val'][env_name])))
        self.env_names = sketch_lengths

        start = time.time()
        a_stats = RunningMeanMax(dimension=9)
        s_stats = RunningMeanMax(dimension=39)
        for env_name in data:
            train_data = self.data['train'][env_name]
            for _traj in train_data:
                for state, action in zip(_traj.states, _traj.actions):
                    a_stats.accumulate(action)
                    s_stats.accumulate(state)

        self.a_mu = a_stats.mean
        a_std = a_stats.max
        zer = np.where(a_std < 0.000001)[0]
        a_std[zer] = 1
        self.a_std = a_std

        self.s_mu = s_stats.mean
        self.s_std = s_stats.max
        logging.info('Compute mean and var cost {} sec'.format(time.time() - start))

    def train_iter(self, batch_size, env_names=None):
        all_train_trajs = []
        env_names = self.env_names if env_names is None else env_names
        for env_name in env_names:
            all_train_trajs += self.data['train'][env_name]
        return self.batch_iter(all_train_trajs, batch_size, shuffle=True, epochs=-1)

    def val_iter(self, batch_size, env_names=None, shuffle=False):
        all_train_trajs = []
        env_names = self.env_names if env_names is None else env_names
        for env_name in env_names:
            all_train_trajs += self.data['val'][env_name]
        return self.batch_iter(all_train_trajs, batch_size, shuffle=shuffle, epochs=1)

    def batch_iter(self, trajs, batch_size, shuffle=True, epochs=-1) -> DictList:
        """
        :param trajs: A list of DictList
        :param batch_size: int
        :param seq_len: int
        :param epochs: int. If -1, then forever
        :return: DictList [bsz, seq_len]
        """
        epoch_iter = range(1, epochs+1) if epochs > 0 else _forever()
        for _ in epoch_iter:
            if shuffle:
                random.shuffle(trajs)

            start_idx = 0
            while start_idx < len(trajs):
                batch = DictList()
                lengths = []
                task_lengths = []
                for _traj in trajs[start_idx: start_idx + batch_size]:
                    lengths.append(len(_traj.actions))
                    task_lengths.append(len(_traj.tasks))
                    _traj.apply(lambda _t: torch.tensor(_t))
                    batch.append(_traj)

                batch.apply(lambda _t: pad_sequence(_t, batch_first=True))
                yield batch, torch.tensor(lengths), torch.tensor(task_lengths)
                start_idx += batch_size


def bucket(val, min, max, nb_bucket):
    each_bucket = (max - min) / nb_bucket
    return (val - min) / each_bucket


def _forever():
    i = 1
    while True:
        yield i
        i += 1
