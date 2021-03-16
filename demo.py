import jacopinpad
from absl import flags
from settings import DATASET_DIR
import os
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_integer('demo_episodes', default=10, help='NB of episodes')


def main():
    for sketch_len in FLAGS.sketch_lengths:
        print('Generating with sketch length {}'.format(sketch_len))
        dataset = jacopinpad.collect_data(nb_episodes=FLAGS.demo_episodes, len_sketch=int(sketch_len), use_dart=True,
                                          permute=True)
        with open(os.path.join(DATASET_DIR, 'jacopinpad_{}.pkl'.format(sketch_len)), 'wb') as f:
            pickle.dump(dataset, f)
