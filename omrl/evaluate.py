"""
Batch evaluation on env
"""
from .visualize import idxpos2tree
from .bots import ModelBot
import numpy as np
from utils import DictList, point_of_change
import torch
from typing import List
from absl import flags


__all__ = ['batch_evaluate', 'free_run_eval']


def step_batch_envs(envs, actions, actives, cuda):
    """ Step a batch of envs. And detect if there are inactive/done envs
    return obss, rewards, dones of the active envs
    """
    assert actions.shape[0] == len(actives)
    active_envs = [envs[i] for i in actives]
    obss = DictList()
    rewards = []
    dones = []
    for action, env in zip(actions, active_envs):
        obs, reward, done, _ = env.step(action.cpu().numpy())
        obss.append(obs)
        rewards.append(reward)
        dones.append(done)

    obss.apply(lambda _t: torch.tensor(_t).float())
    rewards = torch.tensor(rewards).float()
    dones = torch.tensor(dones)

    if cuda:
        obss.apply(lambda _t: _t.cuda())
        rewards = rewards.cuda()
        dones = dones.cuda()

    # Update active
    return obss, rewards, dones


def batch_evaluate(envs: List, bot: ModelBot, cuda, verbose=False) -> List:
    """ Return trajectories after roll out """
    obs = DictList()
    for env in envs:
        obs.append(DictList(env.reset()))
    obs.apply(lambda _t: torch.tensor(_t).float())
    actives = torch.tensor([i for i in range(len(envs))])
    if cuda:
        obs.apply(lambda _t: _t.cuda())
        actives = actives.cuda()

    trajs = [DictList() for _ in range(len(envs))]
    sketchs = obs.sketch.long()[0]
    sketch_lengths = torch.tensor(sketchs.shape, device=sketchs.device)
    mems = bot.init_memory(sketchs.unsqueeze(0).repeat(len(actives), 1),
                           sketch_lengths.repeat(len(actives))) if bot.is_recurrent else None

    # Continue roll out while at least one active
    steps = 0
    while len(actives) > 0:
        if verbose:
            print('active env:', len(actives))
        active_trajs = [trajs[i] for i in actives]
        with torch.no_grad():
            model_outputs = bot.get_action(obs.state, sketchs.unsqueeze(0).repeat(len(actives), 1),
                                           sketch_lengths.repeat(len(actives)), mems)
        actions = model_outputs.actions
        next_obs, rewards, dones = step_batch_envs(envs, actions, actives, cuda)
        transition = DictList({'rewards': rewards})
        transition.update(obs)

        for idx, active_traj in enumerate(active_trajs):
            active_traj.append(transition[idx])
        steps += 1

        # Memory
        next_mems = None
        if bot.is_recurrent:
            next_mems = model_outputs.mems

        # For next step
        un_done_ids = (~dones).nonzero().squeeze(-1)
        obs = next_obs[un_done_ids]
        actives = actives[un_done_ids]
        mems = next_mems[un_done_ids] if next_mems is not None else None

    metric = DictList()
    for traj, env in zip(trajs, envs):
        traj.apply(lambda _tensors: torch.stack(_tensors))
        metric.append({'ret': sum(env.local_score),
                       'succs': traj.rewards.sum().item(),
                       'length': len(traj.rewards)})
    metric.apply(lambda _t: np.mean(_t))
    return metric


def free_run_eval(bot, val_metrics, action_mode='greedy'):
    return val_metrics


def parsing_loop(bot, dataloader, batch_size, cuda):
    bot.eval()
    parsing_metric = {env: DictList() for env in dataloader.env_names}
    for env_name in dataloader.env_names:
        data_iter = dataloader.val_iter(batch_size=batch_size, env_names=[env_name], shuffle=True)
        batch, batch_lens, batch_sketch_lens = data_iter.__next__()
        if cuda:
            batch.apply(lambda _t: _t.cuda())
            batch_lens = batch_lens.cuda()
            batch_sketch_lens = batch_sketch_lens.cuda()
        with torch.no_grad():
            _, extra_info = bot.teacherforcing_batch(batch, batch_lens,
                                                     batch_sketch_lens, recurrence=100)
        for batch_id, (length, sketch_length, ps) in enumerate(zip(batch_lens, batch_sketch_lens, extra_info.p)):
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
                _decoded_subtask = get_subtask_seq(length.item(), subtask=traj.tasks.tolist(),
                                                   use_ids=np.array(preds))
                parsing_metric[env_name] += {'task_acc_thres{}'.format(threshold):
                                                 (_gt_subtask.cpu() == _decoded_subtask.cpu()).tolist()}

            preds = automatic_get_boundaries_peak(ps, bot.nb_slots, nb_boundaries=len(target))
            get_subtask_seq(length.item(), subtask=traj.tasks.tolist(),
                            use_ids=np.array(preds))
            parsing_metric[env_name] += {'task_acc_auto':
                                             (_gt_subtask.cpu() == _decoded_subtask.cpu()).tolist()}

    # print task alignment
    lines = []
    lines.append('tru_ids: {}'.format(target))
    lines.append('dec_ids: {}'.format(preds))
    lines.append(idxpos2tree(pos=ps))
    return parsing_metric, lines


def f1(targets, preds, tolerance=1, with_details=False):
    nb_preds = len(preds)
    nb_targets = len(targets)
    correct_targets = 0
    for tar in targets:
        matched = False
        for pred in preds:
            if tar <= pred + tolerance and tar >= pred - tolerance:
                matched = True
                break

        if matched:
            correct_targets += 1
    recall = correct_targets / nb_targets

    correct_preds = 0
    for pred in preds:
        matched = False
        for tar in targets:
            if pred <= tar + tolerance and pred >= tar - tolerance:
                matched = True
                break
        if matched:
            correct_preds += 1
    pre = correct_preds / nb_preds
    f1 = 2 * pre * recall / (pre + recall) if pre + recall > 0 else 0.
    if with_details:
        return {'f1': f1, 'recall': recall, 'precision': pre}
    else:
        return f1


def get_subtask_seq(action_length, subtask, use_ids):
    task_end_ids = use_ids + 1
    start = 0
    gt = torch.ones(action_length) * subtask[-1]
    for i, end_id in enumerate(task_end_ids):
        gt[start: end_id] = subtask[i]
        start = end_id
    return gt


def get_boundaries(ps, nb_slots, threshold, nb_boundaries=None):
    """ Pick the end of segment as boundaries """
    # Get average p standardized
    ps[0, :-1] = 0
    ps[0, -1] = 1
    p_vals = torch.arange(nb_slots + 1, device=ps.device).flip(0)
    avg_p = (p_vals * ps).sum(-1)
    avg_p = avg_p / (avg_p.max() - avg_p.min())

    # Predicted boundaries as end of segment
    preds = []
    prev = False
    for t in range(len(avg_p)):
        curr = avg_p[t] > threshold
        if not prev and not curr:
            pass
        elif prev and curr:
            pass
        else:
            if prev and not curr:
                preds.append(t - 1)
            prev = curr
    preds.append(len(avg_p) - 1)
    if nb_boundaries is not None:
        preds = preds[::-1][:nb_boundaries][::-1]
    return preds

STEP_SIZE = 0.

def automatic_second_get_boundaries(ps, nb_slots, nb_boundaries, with_details=False):
    """ Pick the end of segment as boundaries """
    # Get average p standardized
    # Get average p standardized
    ps[0, :-1] = 0
    ps[0, -1] = 1
    p_vals = torch.arange(nb_slots + 1, device=ps.device).flip(0)
    avg_p = (p_vals * ps).sum(-1)
    avg_p = avg_p / (avg_p.max() - avg_p.min())

    # diff[i] = avg_p[i+1] - avg[i]
    diff = avg_p[1:] - avg_p[:-1]

    # i+1 is up, if diff[i+1] < 0 and diff[i] > 0
    up_peak_id = [i+1 for i in range(len(diff) - 1) if diff[i] > 0 and diff[i+1] < 0] + [len(diff) - 1]

    # i is down, if diff[i] > 0 and diff[i-1] < 0
    down_peak_id = [0] + [i for i in range(1, len(diff)) if diff[i - 1] < 0 and diff[i] > 0]
    up_peak_val = avg_p[up_peak_id]
    down_peak_val = avg_p[down_peak_id]

    upper = up_peak_val.topk(nb_boundaries).values[-1].item()
    lower = down_peak_val.topk(nb_boundaries, largest=False).values[-1].item()
    final_thres = (upper + lower) / 2
    result = get_boundaries(ps, nb_slots, threshold=final_thres, nb_boundaries=nb_boundaries)
    if not with_details:
        return result
    else:
        return {'final_res': result, 'final_thres': final_thres,
                'upper_thres': upper,
                'lower_thres': lower}


def automatic_topk_diff(ps, nb_slots, nb_boundaries):
    # Get average p standardized
    # Get average p standardized
    ps[0, :-1] = 0
    ps[0, -1] = 1
    p_vals = torch.arange(nb_slots + 1, device=ps.device).flip(0)
    avg_p = (p_vals * ps).sum(-1)
    avg_p = avg_p / (avg_p.max() - avg_p.min())

    # diff[i] = avg_p[i+1] - avg[i]
    avg_p = torch.cat([avg_p, torch.tensor([0], device=avg_p.device)])
    diff = avg_p[1:] - avg_p[:-1]

    # i is up, if diff[i] < 0 and diff[i-1] > 0
    up_peak_id = [i for i in range(1, len(diff)) if diff[i] < 0 and diff[i-1] > 0]
    up_peak_decreased = diff[up_peak_id]

    topK_decreased_values, topK_decreased_id = up_peak_decreased.topk(nb_boundaries,
                                                                      largest=False)
    results = [up_peak_id[idx] for idx in topK_decreased_id]
    results.sort()
    return results


def automatic_get_boundaries_peak(ps, nb_slots, nb_boundaries, with_details=False):
    """ Pick the end of segment as boundaries """
    # Get average p standardized
    # Get average p standardized
    ps[0, :-1] = 0
    ps[0, -1] = 1
    p_vals = torch.arange(nb_slots + 1, device=ps.device).flip(0)
    avg_p = (p_vals * ps).sum(-1)
    avg_p = avg_p / (avg_p.max() - avg_p.min())

    # diff[i] = avgp[i+1] -avgp[i]
    diff = avg_p[1:] - avg_p[:-1]
    up_peak_id = [i for i in range(1, len(diff)) if diff[i] < 0 and diff[i-1] > 0]
    low_peak_id = [i for i in range(1, len(diff)) if diff[i] > 0 and diff[i - 1] < 0]
    up_peak_val = avg_p[up_peak_id]
    low_peak_val = avg_p[low_peak_id]
    upper = up_peak_val.max().item()
    lower = low_peak_val.min().item()
    final_thres = (upper + lower) / 2
    result = get_boundaries(ps, nb_slots, threshold=final_thres, nb_boundaries=nb_boundaries)
    if not with_details:
        return result
    else:
        return {'final_res': result, 'final_thres': final_thres,
                'upper_thres': upper,
                'lower_thres': lower}


def automatic_get_boundaries(ps, nb_slots, nb_boundaries, with_details=False):
    """ Pick the end of segment as boundaries """
    # Get average p standardized
    # Get average p standardized
    ps[0, :-1] = 0
    ps[0, -1] = 1
    p_vals = torch.arange(nb_slots + 1, device=ps.device).flip(0)
    avg_p = (p_vals * ps).sum(-1)
    avg_p = avg_p / (avg_p.max() - avg_p.min())

    search = np.linspace(avg_p.min().item(), avg_p.max().item(), 50)
    for upper in search:
        upper_result = get_boundaries(ps, nb_slots, threshold=upper, nb_boundaries=nb_boundaries)
        if len(upper_result) == nb_boundaries:
            break
    for lower in reversed(search):
        lower_result = get_boundaries(ps, nb_slots, threshold=lower, nb_boundaries=nb_boundaries)
        if len(lower_result) == nb_boundaries:
            break
    final_thres = (upper + lower) / 2
    result = get_boundaries(ps, nb_slots, threshold=final_thres, nb_boundaries=nb_boundaries)
    if not with_details:
        return result
    else:
        return {'final_res': result, 'final_thres': final_thres,
                'upper_thres': upper,
                'lower_thres': lower}

