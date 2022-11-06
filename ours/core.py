import numpy as np
import torch
import jacopinpad
from utils import DictList
from taco.model import ModularPolicy
import gym
from mujoco_py.builder import MujocoException
import time

NEG_INF = -100000000.0
ZERO = 1e-32



def train_batch(
    model: ModularPolicy,
    batch: list,
):
    """Return log probs of a trajectory"""
    start_state, end_state, start, delta, goal, ctx, model_inp = batch

    pred = model(model_inp)

    return torch.nn.functional.mse_loss(pred, end_state)
    

def evaluate_on_env(
    modular_p: ModularPolicy, sketch_length, max_steps_per_sketch, use_sketch_id=False
):
    start = time.time()
    env = gym.make(
        "jacopinpad-v0",
        sketch_length=sketch_length,
        max_steps_per_sketch=max_steps_per_sketch,
    )
    device = next(modular_p.parameters()).device
    modular_p.eval()
    obs = DictList(env.reset())
    modular_p.reset(subtasks=obs.sketch)
    obs.apply(lambda _t: torch.tensor(_t, device=device).float())
    done = False
    traj = DictList()
    try:
        while not done:
            if not use_sketch_id:
                action = modular_p.get_action(obs.state.unsqueeze(0))
            else:
                action = modular_p.get_action(
                    obs.state.unsqueeze(0), sketch_idx=int(obs.sketch_idx.item())
                )
            if action is not None:
                next_obs, reward, done, _ = env.step(action.cpu().numpy()[0])
                transition = {"reward": reward, "action": action, "features": obs.state}
                traj.append(transition)

                obs = DictList(next_obs)
                obs.apply(lambda _t: torch.tensor(_t, device=device).float())
            else:
                done = True
    except MujocoException:
        pass
    end = time.time()
    if "reward" in traj:
        return {
            "succs": np.sum(traj.reward),
            "episode_length": len(traj.reward),
            "ret": sum(env.local_score),
            "runtime": end - start,
        }
    else:
        return {"succs": 0, "episode_length": 0, "ret": 0, "runtime": end - start}
