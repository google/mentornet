# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Visualizes the learned MentorNet.

Run the following command before running the python code.
export PYTHONPATH="$PYTHONPATH:$PWD/code/"
"""
import os
import re
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D  # pylint: disable=unused-import
import numpy as np
import tensorflow as tf
import utils

flags = tf.app.flags
flags.DEFINE_string('model_dir', '', 'Directory to the save training data.')
flags.DEFINE_float('loss_bound', 15,
                   'the upper and lower bound of input loss in the plot.')
flags.DEFINE_list('epoch_ranges', '0,10,20,30,40,50,60,70,80,90,99',
                  'comma-separated list of epoch range to plot.')

FLAGS = flags.FLAGS


def plot_weighting_scheme(model_dir,
                          output_path,
                          loss_bound=15,
                          epoch_ranges=range(0, 100, 10)):
  """Plots the weighting scheme on a 3D graph.

    The axies are
    x: loss,
    y: difference of loss from moving average,
    z: MentorNet Weight,

  Args:
    model_dir: directory where model checkpoints are saved
    output_path: path to save the images
    loss_bound: Integer to which loss should be bound
    epoch_ranges: epochs to plot, represented as percentage from 0 to 99.
  """

  np.set_printoptions(threshold=np.inf)
  np.random.seed(0)

  tf.logging.info('load model from %s', model_dir)

  tf.reset_default_graph()

  x_range = np.arange(-loss_bound, loss_bound, loss_bound/1500.0)
  batch_size = len(x_range)

  input_data_pl = tf.placeholder(
      tf.float32, shape=(batch_size, 4), name='input_data_pl')
  # pylint: disable=undefined-variable
  v = tf.nn.sigmoid(utils.mentornet_nn(input_data_pl), name='v')

  saver = tf.train.Saver()

  with tf.Session() as session:
    if os.path.isdir(model_dir):
      ckpt = tf.train.latest_checkpoint(model_dir)
    else:
      ckpt = model_dir

    if ckpt:
      saver.restore(session, ckpt)

    for t in epoch_ranges:
      loss = np.random.rand(batch_size).astype(np.float32) * loss_bound
      diff = x_range[0:batch_size]
      y = np.zeros(batch_size)
      epoch = np.ones(batch_size) * t

      input_data = np.transpose(np.vstack((loss, diff, y, epoch)))

      v_val = session.run(v, feed_dict={input_data_pl: input_data})

      axis = np.transpose(np.vstack((loss, diff)))
      axis = np.hstack((axis, v_val))

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      ax.scatter(axis[:, 1], axis[:, 0], axis[:, 2], c='r', marker='o', s=10)

      ax.set_ylabel('loss')
      ax.set_xlabel('diff to loss mv')
      ax.set_zlabel('weight')
      ax.set_ylim([0, loss_bound])
      ax.set_xlim([-loss_bound, loss_bound])
      ax.set_zlim([0, 1])
      ax.zaxis.set_major_locator(LinearLocator(9))
      plt.suptitle('epoch={}'.format(t), fontsize=15)
      plt.savefig(os.path.join(output_path, 'epoch_{}.png'.format(t)))


def _make_gif(input_png_path):
  png_filenames = [t for t in os.listdir(input_png_path) if t.endswith('.png')]
  png_filenames = sort_filename(png_filenames)
  out_gif_filename = os.path.join(input_png_path, 'plot.gif')
  images = []
  for filename in png_filenames:
    images.append(imageio.imread(os.path.join(input_png_path, filename)))
  imageio.mimwrite(out_gif_filename, images, duration=0.5, loop=0)


def sort_filename(l):
  """Sorts the given iterable for the text+number file names."""
  convert = lambda text: int(text) if text.isdigit() else text
  alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
  return sorted(l, key=alphanum_key)


def main(_):
  epoch_ranges = [int(t) for t in FLAGS.epoch_ranges]
  if not os.path.isdir(FLAGS.model_dir):
    output_path = os.path.dirname(FLAGS.model_dir)
  else:
    output_path = FLAGS.model_dir

  print output_path
  plot_weighting_scheme(
      model_dir=FLAGS.model_dir,
      output_path=output_path,
      loss_bound=FLAGS.loss_bound,
      epoch_ranges=epoch_ranges)

  _make_gif(output_path)


if __name__ == '__main__':
  tf.app.run()
