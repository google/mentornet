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

"""Evaluates a trained model.

See the README.md file for compilation and running instructions.
"""

import math
import os
import cifar_data_provider
import inception_model
import numpy as np
import resnet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 25, 'The number of images in each batch.')

flags.DEFINE_string('data_dir', '', 'Data dir')

flags.DEFINE_string('dataset_name', 'cifar10', 'cifar10 or cifar100')

flags.DEFINE_string('studentnet', 'resnet101', 'inception or resnet101')

flags.DEFINE_string('master', None, 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '', 'Directory where the results are saved to.')

flags.DEFINE_integer(
    'eval_interval_secs', 600,
    'The frequency, in seconds, with which evaluation is run.')

flags.DEFINE_string('split_name', 'test', """Either 'train' or 'test'.""")

flags.DEFINE_string('output_csv_file', '',
                    'The csv file where the results are saved.')

flags.DEFINE_string('device_id', '0', 'GPU device ID to run the job.')

FLAGS = flags.FLAGS

# Turn this on if there are no log outputs.
tf.logging.set_verbosity(tf.logging.INFO)


def eval_inception():
  """Evalautes the inception model."""
  g = tf.Graph()
  with g.as_default():
    # pylint: disable=line-too-long
    images, one_hot_labels, num_samples, num_of_classes = cifar_data_provider.provide_cifarnet_data(
        FLAGS.dataset_name,
        FLAGS.split_name,
        FLAGS.batch_size,
        dataset_dir=FLAGS.data_dir,
        num_epochs=None)

    # Define the model:
    logits, end_points = inception_model.cifarnet(
        images, num_of_classes, is_training=False, dropout_keep_prob=1.0)
    images.set_shape([FLAGS.batch_size, 32, 32, 3])

    predictions = tf.argmax(end_points['Predictions'], 1)

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_labels, logits=logits)
    total_loss = tf.reduce_mean(total_loss, name='xent')
    slim.summaries.add_scalar_summary(
        total_loss, 'total_loss', print_summary=True)

    # Define the metrics:
    labels = tf.argmax(one_hot_labels, 1)
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': tf.metrics.accuracy(predictions, labels),
    })

    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(
          value, name, prefix='eval', print_summary=True)

    # This ensures that we make a single pass over all of the data.
    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))

    # Limit gpu memory to run train and eval on the same gpu
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.45

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        session_config=config,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


def eval_resnet():
  """Evaluates the resnet model."""
  if not os.path.exists(FLAGS.eval_dir):
    os.makedirs(FLAGS.eval_dir)
  g = tf.Graph()
  with g.as_default():
    # pylint: disable=line-too-long
    images, one_hot_labels, num_samples, num_of_classes = cifar_data_provider.provide_resnet_data(
        FLAGS.dataset_name,
        FLAGS.split_name,
        FLAGS.batch_size,
        dataset_dir=FLAGS.data_dir,
        num_epochs=None)

    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=num_of_classes,
        min_lrn_rate=0.0001,
        lrn_rate=0,
        num_residual_units=9,
        use_bottleneck=False,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer='mom')

    # Define the model:
    images.set_shape([FLAGS.batch_size, 32, 32, 3])
    resnet = resnet_model.ResNet(hps, images, one_hot_labels, mode='test')

    logits = resnet.build_model()

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_labels, logits=logits)
    total_loss = tf.reduce_mean(total_loss, name='xent')

    slim.summaries.add_scalar_summary(
        total_loss, 'total_loss', print_summary=True)

    # Define the metrics:
    predictions = tf.argmax(logits, 1)
    labels = tf.argmax(one_hot_labels, 1)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': tf.metrics.accuracy(predictions, labels),
    })

    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(
          value, name, prefix='eval', print_summary=True)

    # This ensures that we make a single pass over all of the data.
    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


def extract_resnet_features(max_step_run=39000):
  """Not checked provide_resnet_noisy_data dataset might change."""
  g = tf.Graph()

  with g.as_default():
    tf_global_step = tf.train.get_or_create_global_step()

    # pylint: disable=line-too-long
    images, one_hot_labels, num_examples, num_of_classes, clean_labels, image_ids = cifar_data_provider.provide_resnet_noisy_data(
        FLAGS.dataset_name,
        'train',
        FLAGS.batch_size,
        dataset_dir=FLAGS.data_dir)

    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=num_of_classes,
        min_lrn_rate=0.0001,
        lrn_rate=0,
        num_residual_units=9,
        use_bottleneck=False,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer='mom')

    images.set_shape([FLAGS.batch_size, 32, 32, 3])

    # Define the model:
    resnet = resnet_model.ResNet(hps, images, one_hot_labels, mode='train')
    logits = resnet.build_model()

    # Specify the loss function:
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_labels, logits=logits)
    labels = tf.argmax(one_hot_labels, 1)

    loss = tf.reshape(loss, [-1, 1])

    epoch_step = tf.to_int32(
        tf.floor(tf.divide(tf_global_step, max_step_run) * 100))

    ckpt_model = FLAGS.checkpoint_dir

    num_batches = int(math.ceil(num_examples / float(FLAGS.batch_size)))
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        ckpt_model,
        tf.contrib.framework.get_variables_to_restore(include=['.*']))
    outfile = open(FLAGS.output_csv_file, 'w')
    with tf.Session() as sess:
      with slim.queues.QueueRunners(sess):
        init_fn(sess)
        for _ in xrange(num_batches):
          image_ids_val, epoch_step_val, labels_val, loss_val, clean_labels_val = sess.run(
              [image_ids, epoch_step, labels, loss, clean_labels])
          clean_labels_val = np.squeeze(clean_labels_val)
          loss_val = np.squeeze(loss_val)
          image_ids_val = np.squeeze(image_ids_val)
          labels_val = np.squeeze(labels_val)
          for k in range(FLAGS.batch_size):
            outfile.write('{} {} {} {} {:5f}\n'.format(
                int(image_ids_val[k]), epoch_step_val, labels_val[k],
                clean_labels_val[k], loss_val[k]))

    outfile.flush()
    outfile.close()


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_id
  if FLAGS.studentnet == 'resnet101':
    eval_resnet()
  elif FLAGS.studentnet == 'inception':
    eval_inception()
  else:
    tf.logging.error('unknown backbone student network %s', FLAGS.studentnet)


if __name__ == '__main__':
  tf.app.run()
