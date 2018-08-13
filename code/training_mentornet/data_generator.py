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

"""Generates training data for learning/updating MentorNet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import itertools
import os
import pickle
import models
import numpy as np
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('outdir', '', 'Directory to the save training data.')
flags.DEFINE_string('vstar_fn', '', 'the vstar function to use.')
flags.DEFINE_string('vstar_gamma', '', 'the hyper_parameter for the vstar_fn')
flags.DEFINE_integer('sample_size', 100000,
                     'size to of the total generated data set.')

flags.DEFINE_string('input_csv_filename', '', 'input_csv_filename')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def generate_pretrain_defined(vstar_fn, outdir, sample_size):
  """Generates a trainable dataset given a vstar_fn.

  Args:
    vstar_fn: the name of the variable star function to use.
    outdir: directory to save the training data.
    sample_size: size of the sample.
  """
  batch_l = np.concatenate((np.arange(0, 10, 0.1), np.arange(10, 30, 1)))
  batch_diff = np.arange(-5, 5, 0.1)
  batch_y = np.array([0])
  batch_e = np.arange(0, 100, 1)

  data = []
  for t in itertools.product(batch_l, batch_diff, batch_y, batch_e):
    data.append(t)
  data = np.array(data)

  v = vstar_fn(data)
  v = v.reshape([-1, 1])
  data = np.hstack((data, v))

  perm = np.arange(data.shape[0])
  np.random.shuffle(perm)
  data = data[perm[0:min(sample_size, len(perm))],]

  tr_size = int(data.shape[0] * 0.8)

  tr = data[0:tr_size]
  ts = data[(tr_size + 1):data.shape[0]]

  if not os.path.exists(outdir):
    os.makedirs(outdir)

  print('training_shape={} test_shape={}'.format(tr.shape, ts.shape))

  with open(os.path.join(outdir, 'tr.p'), 'wb') as outfile:
    pickle.dump(tr, outfile)

  with open(os.path.join(outdir, 'ts.p'), 'wb') as outfile:
    pickle.dump(ts, outfile)


def generate_data_driven(input_csv_filename,
                         outdir,
                         percentile_range='40,50,60,75,80,90'):
  """Generates a data-driven trainable dataset, given a CSV.

  Refer to README.md for details on how to format the CSV.

  Args:
    input_csv_filename: the path of the CSV file. The csv file format
      0: epoch_percentage
      1: noisy label
      2: clean label
      3: loss
    outdir: directory to save the training data.
    percentile_range: the percentiles used to compute the moving average.
  """
  raw = read_from_csv(input_csv_filename)

  raw = np.array(raw.values())
  dataset_name = os.path.splitext(os.path.basename(input_csv_filename))[0]

  percentile_range = percentile_range.split(',')
  percentile_range = [int(x) for x in percentile_range]

  for percentile in percentile_range:
    percentile = int(percentile)
    p_perncentile = np.percentile(raw[:, 3], percentile)

    v_star = np.float32(raw[:, 1] == raw[:, 2])

    l = raw[:, 3]
    diff = raw[:, 3] - p_perncentile
    # label not used in the current version.
    y = np.array([0] * len(v_star))
    epoch_percentage = raw[:, 0]

    data = np.vstack((l, diff, y, epoch_percentage, v_star))
    data = np.transpose(data)

    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)
    data = data[perm,]

    tr_size = int(data.shape[0] * 0.8)

    tr = data[0:tr_size]
    ts = data[(tr_size + 1):data.shape[0]]

    cur_outdir = os.path.join(
        outdir, '{}_percentile_{}'.format(dataset_name, percentile))
    if not os.path.exists(cur_outdir):
      os.makedirs(cur_outdir)

    print('training_shape={} test_shape={}'.format(tr.shape, ts.shape))
    print(cur_outdir)
    with open(os.path.join(cur_outdir, 'tr.p'), 'wb') as outfile:
      pickle.dump(tr, outfile)

    with open(os.path.join(cur_outdir, 'ts.p'), 'wb') as outfile:
      pickle.dump(ts, outfile)


def read_from_csv(input_csv_file):
  """Reads Data from an input CSV file.

  Args:
    input_csv_file: the path of the CSV file.

  Returns:
    a numpy array with different data at each index:
  """
  data = {}
  with open(input_csv_file, 'r') as csv_file_in:
    reader = csv.reader(csv_file_in)
    for row in reader:
      for (_, cell) in enumerate(row):
        rdata = cell.strip().split(' ')
        rid = rdata[0]
        rdata = [float(t) for t in rdata[1:]]
        data[rid] = rdata
    csv_file_in.close()
  return data


def main(_):
  if FLAGS.vstar_fn == 'data_driven':
    generate_data_driven(FLAGS.input_csv_filename, FLAGS.outdir)
  elif FLAGS.vstar_fn in dir(models):
    generate_pretrain_defined(
        getattr(models, FLAGS.vstar_fn), FLAGS.outdir, FLAGS.sample_size)
  else:
    tf.logging.error('%s is not defined in models.py', FLAGS.vstar_fn)


if __name__ == '__main__':
  tf.app.run()
