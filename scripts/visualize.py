"""
Visuzlizing
"""
import os
from omrl.evaluate import automatic_get_boundaries_peak
import argparse
import torch
import numpy as np
from utils import point_of_change
from utils import DictList
import jacopinpad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import imageio

FPS = 25

parser = argparse.ArgumentParser()
parser.add_argument('--model_ckpt', required=True)
parser.add_argument('--sketch_lengths', required=True, nargs='+')
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--outdir', default='out')
parser.add_argument('--video', action='store_true')
args = parser.parse_args()


def visualize(args):
    os.makedirs(args.outdir, exist_ok=True)
    bot = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    bot.eval()
    sketch_length = int(args.sketch_lengths[0])
    model_name = os.path.dirname(os.path.abspath(args.model_ckpt)).split('/')[-1]
    for episode_id in range(args.episodes):
        demo_traj = jacopinpad.collect_data(1, len_sketch=sketch_length, img_collect=True, permute=False,
                                            use_dart=True)

        # Teacher forcing
        batch = DictList({k: demo_traj[k] for k in ['states', 'actions', 'gt_onsets', 'tasks']})
        batch.apply(lambda _t: torch.tensor(_t))
        batch_lengths = torch.tensor([len(batch.states[0])])
        batch_sketch_lens = torch.tensor([sketch_length])
        with torch.no_grad():
            _, extra_info = bot.teacherforcing_batch(batch, batch_lengths,
                                                     batch_sketch_lens, recurrence=64)

        # Get prediction sorted
        ps = extra_info.p[0]
        ps[0, :-1] = 0
        ps[0, -1] = 1
        p_vals = torch.arange(bot.nb_slots + 1, device=ps.device).flip(0)
        avg_p = (p_vals * ps).sum(-1)
        avg_p = avg_p / (avg_p.max() - avg_p.min())
        p_avg_fig, p_avg_ax = plt.subplots()
        p_avg_ax.plot(avg_p, '--X')
        automatic_results = automatic_get_boundaries_peak(ps, bot.nb_slots, sketch_length, with_details=True)
        final_thres, = p_avg_ax.plot([automatic_results['final_thres']]*len(avg_p), '--r')
        upper_thres, = p_avg_ax.plot([automatic_results['upper_thres']]*len(avg_p), '--y')
        lower_thres, = p_avg_ax.plot([automatic_results['lower_thres']]*len(avg_p), '--b')
        p_avg_ax.legend([final_thres, upper_thres, lower_thres], ['final', 'upper', 'lower'],
                        fontsize=13, loc='upper left')

        preds = [0] + automatic_results['final_res']
        for idx, pred in enumerate(preds):
            img = demo_traj['images'][0][pred]
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.tick_params(axis='x', which='both', bottom=False,
                           top=False, labelbottom=False)
            ax.tick_params(axis='y', which='both', left=False,
                           right=False, labelleft=False)
            fig.savefig(os.path.join(args.outdir, model_name + '_subtask_{}_{}.png'.format(episode_id, idx)),
                        bbox_inches='tight')
            p_avg_ax.plot([pred], [avg_p[pred]], 'r.', markersize=15)

        p_avg_fig.savefig(os.path.join(args.outdir, model_name + '_p_avg_{}.png'.format(episode_id)),
                          bbox_inches='tight')


def video_and_gif(args):
    os.makedirs(args.outdir, exist_ok=True)
    bot = torch.load(args.model_ckpt, map_location=torch.device('cpu'))
    bot.eval()
    sketch_length = int(args.sketch_lengths[0])
    model_name = os.path.dirname(os.path.abspath(args.model_ckpt)).split('/')[-1]
    for episode_id in range(args.episodes):
        demo_traj = jacopinpad.collect_data(1, len_sketch=sketch_length, img_collect=True, permute=False,
                                            use_dart=True)

        # Teacher forcing
        batch = DictList({k: demo_traj[k] for k in ['states', 'actions', 'gt_onsets', 'tasks']})
        batch.apply(lambda _t: torch.tensor(_t))
        batch_lengths = torch.tensor([len(batch.states[0])])
        batch_sketch_lens = torch.tensor([sketch_length])
        with torch.no_grad():
            _, extra_info = bot.teacherforcing_batch(batch, batch_lengths,
                                                     batch_sketch_lens, recurrence=64)

        traj = batch[0]
        gt_subtask = traj.gt_onsets
        tasks = gt_subtask[point_of_change(gt_subtask)]

        # Get prediction sorted
        ps = extra_info.p[0]
        ps[0, :-1] = 0
        ps[0, -1] = 1
        p_vals = torch.arange(bot.nb_slots + 1, device=ps.device).flip(0)
        avg_p = (p_vals * ps).sum(-1)
        avg_p = avg_p / (avg_p.max() - avg_p.min())
        automatic_results = automatic_get_boundaries_peak(ps, bot.nb_slots, sketch_length, with_details=True)
        preds = automatic_results['final_res']

        def get_p_avg_img(time):
            p_avg_fig, p_avg_ax = plt.subplots()
            plt.tight_layout()
            width, height = p_avg_fig.get_size_inches() * p_avg_fig.get_dpi()
            p_avg_ax.plot(avg_p, '--X')
            p_avg_ax.plot([automatic_results['final_thres']]*len(avg_p), 'b',
                          label='final threshold')
            p_avg_ax.plot([time, time], [0, 1], 'r')
            p_avg_ax.legend(fontsize=15, loc='lower left')
            for idx, pred in enumerate(preds):
                p_avg_ax.plot([pred], [avg_p[pred]], 'rX')

            canvas = FigureCanvas(p_avg_fig)
            canvas.draw()  # draw the canvas, cache the renderer
            np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            plt.close(p_avg_fig)
            return np_image

        print()
        sketch_id = 0
        vid_frames = []
        for t, raw_img in tqdm(enumerate(demo_traj['images'][0])):
            img = Image.fromarray(raw_img)
            p_avg_img = Image.fromarray(get_p_avg_img(t))
            task = tasks[sketch_id]
            draw = ImageDraw.Draw(img)
            fonts_path = os.path.join(os.path.dirname(__file__), 'fonts')
            font = ImageFont.truetype(os.path.join(fonts_path, 'sans_serif.ttf'), 50)
            draw.text((20, 0), "Press {}".format(task), (0, 0, 0), font=font)

            ratio = p_avg_img.height / img.height
            new_height = p_avg_img.height
            new_width = int(img.width * ratio)
            img = img.resize((new_width, new_height))
            final = get_concat_h(p_avg_img, img)
            vid_frames.append(final)
            if t in preds:
                # Do extra 1 second frames
                for _ in range(2 * FPS):
                    vid_frames.append(final)
                sketch_id += 1

        # Produce video
        print('Producing videos...')
        videodims = (vid_frames[0].width, vid_frames[0].height)
        video = cv2.VideoWriter(os.path.join(args.outdir, model_name + "_{}.mp4".format(episode_id)),
                                0x7634706d, FPS, videodims)
        for frame in vid_frames:
            video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
        video.release()

        # Produce Gif
        print('Producing GIF...')
        gif_path = os.path.join(args.outdir, model_name + "_{}.gif".format(episode_id))
        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
            for frame in tqdm(vid_frames[::5]):
                writer.append_data(np.array(frame))


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


if __name__ == '__main__':
    if args.video:
        video_and_gif(args)
    else:
        visualize(args)
