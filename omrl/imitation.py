"""
imitation
"""
from absl import flags, logging
import torch
import time
from .bots import ModelBot, make
from utils import DictList, logging_metrics
from tensorboardX import SummaryWriter
import numpy as np
from data import Dataloader
import os
import gym
from .evaluate import batch_evaluate, parsing_loop

FLAGS = flags.FLAGS

# Misc
flags.DEFINE_bool('il_demo_from_model', default=False, help='Whether to use demo from bot')
flags.DEFINE_integer('il_train_steps', default=50000, help='Trainig steps')
flags.DEFINE_integer('il_eval_freq', default=20, help='Evaluation frequency')
flags.DEFINE_integer('il_save_freq', default=200, help='Save freq')

# Data
flags.DEFINE_float('il_val_ratio', default=0.2,
                   help='Validation')
flags.DEFINE_integer('il_batch_size', default=128,
                     help='batch size')

# Optimize
flags.DEFINE_integer('il_recurrence', default=30, help='bptt length')
flags.DEFINE_float('il_lr', default=5e-4, help="Learning rate")
flags.DEFINE_float('il_clip', default=0.2, help='RNN clip')


def run_batch(batch: DictList,
              batch_lengths,
              sketch_lengths,
              bot: ModelBot,
              mode='train') \
        -> DictList:
    """
    :param batch: DictList object [bsz, seqlen]
    :param bot:  A model Bot
    :param mode: 'train' or 'eval'
    :return:
        stats: A DictList of bsz, mem_size
    """
    bsz, seqlen = batch.actions.shape[0], batch.actions.shape[1]
    sketchs = batch.tasks
    final_outputs = DictList({})
    mems = None
    if bot.is_recurrent:
        mems = bot.init_memory(sketchs, sketch_lengths)

    for t in range(seqlen):
        final_output = DictList({})
        model_output = bot.forward(batch.states[:, t], sketchs, sketch_lengths, mems)
        logprobs = model_output.dist.log_prob(batch.actions[:, t].float())
        if 'log_end' in model_output:
            # p_end + (1 - pend) action_prob
            log_no_end_term = model_output.log_no_end + logprobs
            logprobs = torch.logsumexp(torch.stack([model_output.log_end, log_no_end_term], dim=-1), dim=-1)
            final_output.log_end = model_output.log_end
        final_output.logprobs = logprobs
        final_outputs.append(final_output)

        # Update memory
        next_mems = None
        if bot.is_recurrent:
            next_mems = model_output.mems
            if (t+1) % FLAGS.il_recurrence == 0 and mode =='train':
                next_mems = next_mems.detach()
        mems = next_mems

    # Stack on time dim
    final_outputs.apply(lambda _tensors: torch.stack(_tensors, dim=1))
    sequence_mask = torch.arange(batch_lengths.max().item(),
                                 device=batch_lengths.device)[None, :] < batch_lengths[:, None]
    final_outputs.loss = -final_outputs.logprobs
    if 'log_end' in final_outputs:
        batch_ids = torch.arange(bsz, device=batch.states.device)
        final_outputs.loss[batch_ids, batch_lengths - 1] = final_outputs.log_end[batch_ids, batch_lengths - 1]
    final_outputs.apply(lambda _t: _t.masked_fill(~sequence_mask, 0.))
    return final_outputs


def evaluate_on_envs(bot, dataloader):
    val_metrics = {}
    bot.eval()
    envs = dataloader.env_names
    for env_name in envs:
        val_iter = dataloader.val_iter(batch_size=FLAGS.il_batch_size,
                                       env_names=[env_name], shuffle=True)
        output = DictList({})
        total_lengths = 0
        for batch, batch_lens, batch_sketch_lens in val_iter:
            if FLAGS.cuda:
                batch.apply(lambda _t: _t.cuda())
                batch_lens = batch_lens.cuda()
                batch_sketch_lens = batch_sketch_lens.cuda()

            # Initialize memory
            with torch.no_grad():
                #batch_results = run_batch(batch, batch_lens, batch_sketch_lens, bot, mode='val')
                start = time.time()
                batch_results, _ = bot.teacherforcing_batch(batch, batch_lens,
                                                            batch_sketch_lens, recurrence=FLAGS.il_recurrence)
                end = time.time()
                print('batch time', end - start)
            batch_results.apply(lambda _t: _t.sum().item())
            output.append(batch_results)
            total_lengths += batch_lens.sum().item()
            if FLAGS.debug:
                break
        output.apply(lambda _t: torch.tensor(_t).sum().item() / total_lengths)
        val_metrics[env_name] = {k: v for k, v in output.items()}

    # Parsing
    if 'om' in FLAGS.arch:
        with torch.no_grad():
            parsing_stats, parsing_lines = parsing_loop(bot, dataloader=dataloader,
                                                        batch_size=FLAGS.il_batch_size,
                                                        cuda=FLAGS.cuda)
        for env_name in parsing_stats:
            parsing_stats[env_name].apply(lambda _t: np.mean(_t))
            val_metrics[env_name].update(parsing_stats[env_name])
        logging.info('Get parsing result')
        logging.info('\n' + '\n'.join(parsing_lines))

    # evaluate on free run env
    #if not FLAGS.debug:
    #    for sketch_length in val_metrics:
    #        envs = [gym.make('jacopinpad-v0', sketch_length=sketch_length,
    #                         max_steps_per_sketch=FLAGS.max_steps_per_sketch)
    #                for _ in range(FLAGS.eval_episodes)]
    #        with torch.no_grad():
    #            free_run_metric = batch_evaluate(envs=envs, bot=bot, cuda=FLAGS.cuda)
    #        val_metrics[sketch_length].update(free_run_metric)
    return val_metrics


def main_loop(bot, dataloader, opt, training_folder, test_dataloader=None):
    # Prepare
    train_steps = 0
    writer = SummaryWriter(training_folder)
    train_iter = dataloader.train_iter(batch_size=FLAGS.il_batch_size)
    nb_frames = 0
    train_stats = DictList()
    curr_best = 100000
    while True:
        if train_steps > FLAGS.il_train_steps:
            logging.info('Reaching maximum steps')
            break

        if train_steps % FLAGS.il_save_freq == 0:
            with open(os.path.join(training_folder, 'bot{}.pkl'.format(train_steps)), 'wb') as f:
                torch.save(bot, f)

        if train_steps % FLAGS.il_eval_freq == 0:
            # testing on valid
            val_metrics = evaluate_on_envs(bot, dataloader)
            logging_metrics(nb_frames, train_steps, val_metrics, writer, prefix='val')

            # testing on test env
            if test_dataloader is not None:
                test_metrics = evaluate_on_envs(bot, test_dataloader)
                logging_metrics(nb_frames, train_steps, test_metrics, writer, prefix='test')

            avg_loss = [val_metrics[env_name]['loss'] for env_name in val_metrics]
            avg_loss = np.mean(avg_loss)

            if avg_loss < curr_best:
                curr_best = avg_loss
                logging.info('Save Best with loss: {}'.format(avg_loss))

                # Save the checkpoint
                with open(os.path.join(training_folder, 'bot_best.pkl'), 'wb') as f:
                    torch.save(bot, f)

        # Forward/Backward
        bot.train()
        train_batch, train_lengths, train_sketch_lengths = train_iter.__next__()
        if FLAGS.cuda:
            train_batch.apply(lambda _t: _t.cuda())
            train_lengths = train_lengths.cuda()
            train_sketch_lengths = train_sketch_lengths.cuda()

        start = time.time()
        #train_batch_res = run_batch(train_batch, train_lengths, train_sketch_lengths, bot)
        train_batch_res, _ = bot.teacherforcing_batch(train_batch, train_lengths, train_sketch_lengths,
                                                      recurrence=FLAGS.il_recurrence)
        train_batch_res.apply(lambda _t: _t.sum() / train_lengths.sum())
        batch_time = time.time() - start
        loss = train_batch_res.loss
        opt.zero_grad()
        loss.backward()
        params = [p for p in bot.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters=params, max_norm=FLAGS.il_clip)
        opt.step()
        train_steps += 1
        nb_frames += train_lengths.sum().item()
        fps = train_lengths.sum().item() / batch_time

        stats = DictList()
        stats.grad_norm = grad_norm
        stats.loss = train_batch_res.loss.detach()
        stats.fps = torch.tensor(fps)
        train_stats.append(stats)

        if train_steps % FLAGS.il_eval_freq == 0:
            train_stats.apply(lambda _tensors: torch.stack(_tensors).mean().item())
            logger_str = ['[TRAIN] steps={}'.format(train_steps)]
            for k, v in train_stats.items():
                logger_str.append("{}: {:.4f}".format(k, v))
                writer.add_scalar('train/' + k, v, global_step=nb_frames)
            logging.info('\t'.join(logger_str))
            train_stats = DictList()
            writer.flush()


def run(training_folder):
    logging.info('Start IL...')
    dataloader = Dataloader(FLAGS.sketch_lengths, FLAGS.il_val_ratio)
    bot = make(input_dim=39,
               action_dim=9,
               arch=FLAGS.arch,
               hidden_size=FLAGS.hidden_size,
               nb_slots=FLAGS.nb_slots,
               env_arch=FLAGS.env_arch,
               dataloader=dataloader)
    if FLAGS.cuda:
        bot = bot.cuda()
    params = [p for p in bot.parameters()]
    opt = torch.optim.Adam(params, lr=FLAGS.il_lr)

    # test dataloader
    test_sketch_lengths = set(FLAGS.test_sketch_lengths) - set(FLAGS.sketch_lengths)
    test_dataloader = None if len(test_sketch_lengths) == 0 else Dataloader(test_sketch_lengths, FLAGS.il_val_ratio)

    try:
        main_loop(bot, dataloader, opt, training_folder, test_dataloader)
    except KeyboardInterrupt:
        pass
