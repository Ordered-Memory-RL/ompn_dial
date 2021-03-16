"""
Run Omdec and analyze structure
"""
import os
from omrl.evaluate import f1, get_subtask_seq, get_boundaries, automatic_get_boundaries, automatic_get_boundaries_peak
import argparse
import torch
import numpy as np
from utils import point_of_change
from data import Dataloader
from utils import DictList
from omrl.bots.omdec import OMdecBase
from taco.model import ModularPolicy
from taco.core import teacherforce_batch as taco_decode
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', required=True)
parser.add_argument('--sketch_lengths', required=True, nargs='+')
parser.add_argument('--episodes', type=int, default=500)
args = parser.parse_args()

def taco_eval(bot, args):
    parsing_metric = {}
    dataloader = Dataloader(args.sketch_lengths, 0.99)
    bot.eval()
    for sketch_len in dataloader.env_names:
        parsing_metric[sketch_len] = DictList()
        data_iter = dataloader.val_iter(args.episodes, shuffle=True, env_names=[sketch_len])
        batch, batch_lens, batch_sketch_lens = data_iter.__next__()
        with torch.no_grad():
            parsing_res, _ = taco_decode(bot, trajs=batch, lengths=batch_lens,
                                         subtask_lengths=batch_sketch_lens,
                                         dropout_p=0., decode=True)
        parsing_metric[sketch_len].append(parsing_res)
        parsing_metric[sketch_len].apply(lambda _t: _t[0].item())
    return parsing_metric

def ompn_eval(bot, args):
    parsing_metric = {}
    dataloader = Dataloader(args.sketch_lengths, 0.99)
    bot.eval()
    for sketch_len in dataloader.env_names:
        parsing_metric[sketch_len] = DictList()
        data_iter = dataloader.val_iter(args.episodes, shuffle=True, env_names=[sketch_len])
        batch, batch_lens, batch_sketch_lens = data_iter.__next__()
        with torch.no_grad():
            _, extra_info = bot.teacherforcing_batch(batch, batch_lens,
                                                     batch_sketch_lens, recurrence=64)

        for batch_id, (length, sketch_length, ps) in tqdm(enumerate(zip(batch_lens, batch_sketch_lens, extra_info.p))):
            traj = batch[batch_id]
            traj = traj[:length]
            _gt_subtask = traj.gt_onsets
            target = point_of_change(_gt_subtask)

            # Get prediction sorted
            ps = ps[:length]
            ps[0, :-1] = 0
            ps[0, -1] = 1
            for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
                preds = get_boundaries(ps, bot.nb_slots, threshold=threshold, nb_boundaries=len(target))
                #parsing_metric[sketch_len].append({'f1_tol{}_thres{}'.format(tol, threshold):
                #                                       f1(target, preds, tol) for tol in [0, 1, 2]})
                _decoded_subtask = get_subtask_seq(length.item(), subtask=traj.tasks.tolist(),
                                                   use_ids=np.array(preds))
                parsing_metric[sketch_len] += {'task_acc_thres{}'.format(threshold):
                                                   (_gt_subtask.cpu() == _decoded_subtask.cpu()).tolist()}

            preds = automatic_get_boundaries_peak(ps, bot.nb_slots, nb_boundaries=len(target))
            _decoded_subtask = get_subtask_seq(length.item(), subtask=traj.tasks.tolist(),
                                               use_ids=np.array(preds))
            parsing_metric[sketch_len] += {'task_acc_auto':
                                               (_gt_subtask.cpu() == _decoded_subtask.cpu()).tolist()}

        parsing_metric[sketch_len].apply(lambda _t: np.mean(_t))

    return parsing_metric


if __name__ == '__main__':
    bot = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    bot.eval()
    model_name = os.path.dirname(os.path.abspath(args.model_ckpt)).split('/')[-1]
    if isinstance(bot, OMdecBase):
        results = ompn_eval(bot, args)
    elif isinstance(bot, ModularPolicy):
        results = taco_eval(bot, args)
    print('######### {} ###########'.format(model_name))
    for sketch_len, metric in results.items():
        print('length', sketch_len)
        print('\t'.join(["{}: {:.4f}".format(k, v) for k,v in metric.items()]))
