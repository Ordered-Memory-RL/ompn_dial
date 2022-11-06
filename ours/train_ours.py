import random
from ours.model import ForwardModel
from settings import DATASET_DIR
from utils import DictList, point_of_change, logging_metrics
from data import Dataloader
from ours.core import train_batch, evaluate_on_env
import os
import time
from tensorboardX import SummaryWriter
import numpy as np
import torch
from absl import logging, flags
import webdataset as wds
import torch.nn.functional as F
from torchinfo import summary
import wandb

wandb.init()

DEBUG = False
if not DEBUG:
    FLAGS = flags.FLAGS
    flags.DEFINE_float("ours_lr", default=0.001, help="ours learning rate")
    flags.DEFINE_integer("ours_batch_size", default=32, help="ours batch size")
    flags.DEFINE_integer("ours_train_steps", default=40000, help="ours train steps")
    flags.DEFINE_integer("ours_eval_freq", default=100, help="ours eval freq")
    flags.DEFINE_float("ours_dropout", default=0.3, help="Initial dropout rate")
    flags.DEFINE_float(
        "ours_do_decay_ratio", default=0.8, help="use ratio * train_steps to decay"
    )


class DropoutScheduler:
    def __init__(self):
        self.init_do = FLAGS.ours_dropout
        self.decay_steps = int(FLAGS.ours_do_decay_ratio * FLAGS.ours_train_steps)
        self.min_val = 0.0
        self.steps = 0

    @property
    def dropout_p(self):
        return max(
            (self.decay_steps - self.steps) / self.decay_steps * self.init_do,
            self.min_val,
        )

    def step(self):
        self.steps += 1


def evaluate_loop(batch, model):
    # Testing
    model.eval()
    return train_batch(model, batch)


def get_dataloader(kind):
    pkl_name = os.path.join(
        # DATASET_DIR, f"jacopinpad_{FLAGS.sketch_lengths}_{kind}.pkl"
        DATASET_DIR,
        f"jacopinpad_{4}_{kind}.tar",
    )

    def jump_sample(sample):
        # return sample
        start = random.randint(0, len(sample["states.pyd"]) - 11)
        end = random.randint(0, 10)
        ret = {}
        tasks = torch.from_numpy(sample["tasks.pyd"])

        ret["start"] = start

        ret["start_state"] = sample["states.pyd"][start]
        ret["delta"] = end
        ret["goal"] = F.one_hot(tasks, num_classes=10)
        ret["ctx"] = ret["goal"].clone()

        ret["model_inp"] = np.concatenate(
            (
                ret["start_state"],
                ret["goal"].reshape(-1),
                ret["ctx"].reshape(-1),
                np.array([ret["delta"]]),
            ), dtype=np.float32
        )

        ret["end_state"] = sample["states.pyd"][start + end].float()
        for i, change in enumerate(sample["points_of_change.pyd"]):
            if change > start:
                ret["ctx"][i] = 0

        return ret

    dataset = wds.DataPipeline(
        wds.SimpleShardList(pkl_name),
        wds.tarfile_to_samples(),
        wds.decode(),
        wds.map(jump_sample),
        wds.to_tuple(
            "start_state", "end_state", "start", "delta", "goal", "ctx", "model_inp"
        ),
        wds.batched(FLAGS.ours_batch_size if not DEBUG else 16),
    )
    dataloader = (
        wds.WebLoader(dataset, num_workers=4, batch_size=None)
        .unbatched()
        .shuffle(1000)
        .batched(FLAGS.ours_batch_size if not DEBUG else 16)
    )
    return dataloader


def main(training_folder):
    logging.info("start ours...")
    dataloader = get_dataloader("train")
    val_dataloader = get_dataloader("val")
    # dataloader = Dataloader(FLAGS.sketch_lengths, 0.2)
    model = ForwardModel()
    summary(model)
    if FLAGS.cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.ours_lr)

    train_steps = 0
    train_iter = iter(dataloader)
    val_iter = iter(val_dataloader)
    nb_frames = 0
    curr_best = np.inf

    while True:
        if train_steps > FLAGS.ours_train_steps:
            logging.info("Reaching maximum steps")
            break

        if train_steps % FLAGS.ours_eval_freq == 0:
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(dataloader)
                val_batch = next(val_iter)

            val_loss = evaluate_loop(val_batch, model)
            wandb.log({'val/loss': val_loss})

            if val_loss < curr_best:
                curr_best = val_loss
                logging.info("Save Best with loss: {}".format(val_loss))
                # Save the checkpoint
                with open(os.path.join(training_folder, "bot_best.pkl"), "wb") as f:
                    torch.save(model, f)

        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(dataloader)
            batch = next(train_iter)

        if FLAGS.cuda:
            batch.apply(lambda _t: _t.cuda())

        start = time.time()
        loss = train_batch(model, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_steps += 1
        nb_frames += FLAGS.ours_batch_size
        end = time.time()
        fps = FLAGS.ours_batch_size / (end - start)
        wandb.log({'train/loss': loss, 'fps': fps})

