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

"""Trains MentorNet models.

See the README.md file for compilation and running instructions.
"""

import os
import time
import cifar_data_provider
import inception_model
import numpy as np
import resnet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

flags.DEFINE_string('master', None, 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('data_dir', '', 'Data dir')

flags.DEFINE_string('train_log_dir', '', 'Directory to the save trained model.')

flags.DEFINE_string('dataset_name', 'cifar10', 'cifar10 or cifar100')

flags.DEFINE_string('studentnet', 'resnet101', 'inception or resnet101')

flags.DEFINE_float('learning_rate', 0.1, 'The learning rate')
flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'learning rate decay factor')

flags.DEFINE_float('num_epochs_per_decay', 50,
                   'Number of epochs after which learning rate decays.')

flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')

flags.DEFINE_integer(
    'save_interval_secs', 1200,
    'The frequency with which the model is saved, in seconds.')

flags.DEFINE_integer('max_number_of_steps', 39000,
                     'The maximum number of gradient steps.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

flags.DEFINE_string('device_id', '0', 'GPU device ID to run the job.')

# Learned MentorNet location
flags.DEFINE_string('trained_mentornet_dir', '',
                    'Directory where to find the trained MentorNet model.')

# Hyper-parameters for MentorNet
flags.DEFINE_float('loss_p_percentile', 0.7, 'p-percentile used to compute'
                   'the loss moving average.')

flags.DEFINE_integer('burn_in_epoch', 0, 'Number of first epochs to perform'
                     'burn-in. In the burn-in period, every sample has a'
                     'fixed 1.0 weight.')

flags.DEFINE_bool('fixed_epoch_after_burn_in', False, 'Whether to use the fixed'
                  'epoch as the MentorNet input feature after the burn-in'
                  'period. Set True for MentorNet DD.')

flags.DEFINE_float('loss_moving_average_decay', 0.5, 'Decay factor used in'
                   'moving average.')

flags.DEFINE_list('example_dropout_rates', '0.5, 17, 0.05, 78, 1.0, 5',
                  'Comma-separated list indicating the example drop-out rate'
                  'for the total of 100 epochs. The format is'
                  '[dropout rate, epoch_num]+, the piecewise drop-out rate from'
                  ' boundaries and values. The sum of epoch_num is 100.')

FLAGS = flags.FLAGS

# Turn this on if there are no log outputs
tf.logging.set_verbosity(tf.logging.INFO)


def resnet_train_step(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.

  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments.

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  Raises:
    ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
  """
  start_time = time.time()

  total_loss = tf.get_collection('total_loss')[0]

  _, np_global_step, total_loss_val = sess.run(
      [train_op, global_step, total_loss])

  time_elapsed = time.time() - start_time

  if 'should_log' in train_step_kwargs:
    if sess.run(train_step_kwargs['should_log']):
      tf.logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                      np_global_step, total_loss_val, time_elapsed)

  if 'should_stop' in train_step_kwargs:
    should_stop = sess.run(train_step_kwargs['should_stop'])
  else:
    should_stop = False
  return total_loss, should_stop


def train_resnet_mentornet(max_step_run):
  """Trains the mentornet with the student resnet model.

  Args:
    max_step_run: The maximum number of gradient steps.
  """
  if not os.path.exists(FLAGS.train_log_dir):
    os.makedirs(FLAGS.train_log_dir)
  g = tf.Graph()

  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      tf_global_step = tf.train.get_or_create_global_step()

      # pylint: disable=line-too-long
      images, one_hot_labels, num_samples_per_epoch, num_of_classes = cifar_data_provider.provide_resnet_data(
          FLAGS.dataset_name,
          'train',
          FLAGS.batch_size,
          dataset_dir=FLAGS.data_dir)

      hps = resnet_model.HParams(
          batch_size=FLAGS.batch_size,
          num_classes=num_of_classes,
          min_lrn_rate=0.0001,
          lrn_rate=FLAGS.learning_rate,
          num_residual_units=9,
          use_bottleneck=False,
          weight_decay_rate=0.0002,
          relu_leakiness=0.1,
          optimizer='mom')

      images.set_shape([FLAGS.batch_size, 32, 32, 3])
      tf.logging.info('num_of_example=%s', num_samples_per_epoch)

      # Define the model:
      resnet = resnet_model.ResNet(hps, images, one_hot_labels, mode='train')
      logits = resnet.build_model()

      # Specify the loss function:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=one_hot_labels, logits=logits)

      dropout_rates = utils.parse_dropout_rate_list(FLAGS.example_dropout_rates)
      example_dropout_rates = tf.convert_to_tensor(
          dropout_rates, np.float32, name='example_dropout_rates')

      loss_p_percentile = tf.convert_to_tensor(
          np.array([FLAGS.loss_p_percentile] * 100),
          np.float32,
          name='loss_p_percentile')

      loss = tf.reshape(loss, [-1, 1])

      epoch_step = tf.to_int32(
          tf.floor(tf.divide(tf_global_step, max_step_run) * 100))

      zero_labels = tf.zeros([tf.shape(loss)[0], 1], tf.float32)

      v = utils.mentornet(
          epoch_step,
          loss,
          zero_labels,
          loss_p_percentile,
          example_dropout_rates,
          burn_in_epoch=FLAGS.burn_in_epoch,
          fixed_epoch_after_burn_in=FLAGS.fixed_epoch_after_burn_in,
          loss_moving_average_decay=FLAGS.loss_moving_average_decay)

      tf.stop_gradient(v)

      # Log data utilization
      data_util = utils.summarize_data_utilization(v, tf_global_step,
                                                   FLAGS.batch_size)
      decay_loss = resnet.decay()
      weighted_loss_vector = tf.multiply(loss, v)

      weighted_loss = tf.reduce_mean(weighted_loss_vector)

      slim.summaries.add_scalar_summary(
          tf.reduce_mean(loss), 'mentornet/orig_loss')
      slim.summaries.add_scalar_summary(weighted_loss,
                                        'mentornet/weighted_loss')

      # Normalize the decay loss based on v
      weighed_decay_loss = decay_loss * (tf.reduce_sum(v) / FLAGS.batch_size)

      weighted_total_loss = weighted_loss + weighed_decay_loss

      slim.summaries.add_scalar_summary(weighted_total_loss,
                                        'mentornet/total_loss')

      slim.summaries.add_scalar_summary(weighted_total_loss, 'total_loss')
      tf.add_to_collection('total_loss', weighted_total_loss)

      boundaries = [19531, 25000, 30000]
      values = [FLAGS.learning_rate * t for t in [1, 0.1, 0.01, 0.001]]
      lr = tf.train.piecewise_constant(tf_global_step, boundaries, values)
      slim.summaries.add_scalar_summary(lr, 'learning_rate')

      # Specify the optimization scheme:
      with tf.control_dependencies([weighted_total_loss, data_util]):
        # Set up training.
        trainable_variables = tf.trainable_variables()
        trainable_variables = tf.contrib.framework.filter_variables(
            trainable_variables, exclude_patterns=['mentornet'])

        grads = tf.gradients(weighted_total_loss, trainable_variables)
        optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9)

        apply_op = optimizer.apply_gradients(
            zip(grads, trainable_variables),
            global_step=tf_global_step,
            name='train_step')

        train_ops = [apply_op] + resnet.extra_train_ops
        train_op = tf.group(*train_ops)

      # Parameter restore setup
      if FLAGS.trained_mentornet_dir is not None:
        ckpt_model = FLAGS.trained_mentornet_dir
        if os.path.isdir(FLAGS.trained_mentornet_dir):
          ckpt_model = tf.train.latest_checkpoint(ckpt_model)

        # Fix the mentornet parameters
        variables_to_restore = slim.get_variables_to_restore(
            # TODO(lujiang): mentornet_inputs or mentor_inputs?
            include=['mentornet', 'mentornet_inputs'])
        iassign_op1, ifeed_dict1 = tf.contrib.framework.assign_from_checkpoint(
            ckpt_model, variables_to_restore)

        # Create an initial assignment function.
        def init_assign_fn(sess):
          tf.logging.info('Restore using customer initializer %s', '.' * 10)
          sess.run(iassign_op1, ifeed_dict1)
      else:
        init_assign_fn = None

      tf.logging.info('-' * 20 + 'MentorNet' + '-' * 20)
      tf.logging.info('loaded pretrained mentornet from %s', ckpt_model)
      tf.logging.info('loss_p_percentile=%3f', FLAGS.loss_p_percentile)
      tf.logging.info('burn_in_epoch=%d', FLAGS.burn_in_epoch)
      tf.logging.info('fixed_epoch_after_burn_in=%s',
                      FLAGS.fixed_epoch_after_burn_in)
      tf.logging.info('loss_moving_average_decay=%3f',
                      FLAGS.loss_moving_average_decay)
      tf.logging.info('example_dropout_rates %s', ','.join(
          str(t) for t in dropout_rates))
      tf.logging.info('-' * 20)

      saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=24)

      # Run training.
      slim.learning.train(
          train_op=train_op,
          train_step_fn=resnet_train_step,
          logdir=FLAGS.train_log_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          saver=saver,
          number_of_steps=max_step_run,
          init_fn=init_assign_fn,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


def train_inception_mentornet(max_step_run):
  """Trains the mentornet with the student inception model.

  Args:
    max_step_run: The maximum number of gradient steps.
  """
  if not os.path.exists(FLAGS.train_log_dir):
    os.makedirs(FLAGS.train_log_dir)
  g = tf.Graph()

  with g.as_default():
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      config = tf.ConfigProto()
      # limit gpu memory to run train and eval on the same gpu
      config.gpu_options.per_process_gpu_memory_fraction = 0.45

      tf_global_step = tf.train.get_or_create_global_step()

      # pylint: disable=line-too-long
      images, one_hot_labels, num_samples_per_epoch, num_of_classes = cifar_data_provider.provide_cifarnet_data(
          FLAGS.dataset_name,
          'train',
          FLAGS.batch_size,
          dataset_dir=FLAGS.data_dir)

      images.set_shape([FLAGS.batch_size, 32, 32, 3])
      tf.logging.info('num_of_example=%s', num_samples_per_epoch)

      # Define the model:
      with slim.arg_scope(
          inception_model.cifarnet_arg_scope(weight_decay=0.004)):
        logits, _ = inception_model.cifarnet(
            images, num_of_classes, is_training=True, dropout_keep_prob=0.8)

      # Specify the loss function:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=one_hot_labels, logits=logits)

      dropout_rates = utils.parse_dropout_rate_list(FLAGS.example_dropout_rates)
      example_dropout_rates = tf.convert_to_tensor(
          dropout_rates, np.float32, name='example_dropout_rates')

      loss_p_percentile = tf.convert_to_tensor(
          np.array([FLAGS.loss_p_percentile] * 100),
          np.float32,
          name='loss_p_percentile')

      epoch_step = tf.to_int32(
          tf.floor(tf.divide(tf_global_step, max_step_run) * 100))

      zero_labels = tf.zeros([tf.shape(loss)[0], 1], tf.float32)

      loss = tf.reshape(loss, [-1, 1])

      v = utils.mentornet(
          epoch_step,
          loss,
          zero_labels,
          loss_p_percentile,
          example_dropout_rates,
          burn_in_epoch=FLAGS.burn_in_epoch,
          fixed_epoch_after_burn_in=FLAGS.fixed_epoch_after_burn_in,
          loss_moving_average_decay=FLAGS.loss_moving_average_decay)

      tf.stop_gradient(v)

      # log data utilization
      data_util = utils.summarize_data_utilization(v, tf_global_step,
                                                   FLAGS.batch_size)

      weighted_loss_vector = tf.multiply(loss, v)

      weighted_loss = tf.reduce_mean(weighted_loss_vector)

      slim.summaries.add_scalar_summary(
          tf.reduce_mean(loss), 'mentornet/orig_loss')
      slim.summaries.add_scalar_summary(weighted_loss,
                                        'mentornet/weighted_loss')

      # normalize the decay loss based on v
      weighed_decay_loss = 0
      weighted_total_loss = weighted_loss + weighed_decay_loss

      slim.summaries.add_scalar_summary(weighted_total_loss,
                                        'mentornet/total_loss')

      slim.summaries.add_scalar_summary(weighted_total_loss, 'total_loss')
      tf.add_to_collection('total_loss', weighted_total_loss)

      decay_steps = int(
          num_samples_per_epoch / FLAGS.batch_size * FLAGS.num_epochs_per_decay)

      lr = tf.train.exponential_decay(
          FLAGS.learning_rate,
          tf_global_step,
          decay_steps,
          FLAGS.learning_rate_decay_factor,
          staircase=True)
      slim.summaries.add_scalar_summary(lr, 'learning_rate', print_summary=True)

      with tf.control_dependencies([weighted_total_loss, data_util]):
        # Set up training.
        trainable_variables = tf.trainable_variables()
        trainable_variables = tf.contrib.framework.filter_variables(
            trainable_variables, exclude_patterns=['mentornet'])

        # Specify the optimization scheme:
        optimizer = tf.train.GradientDescentOptimizer(lr)
        train_op = slim.learning.create_train_op(
            weighted_total_loss,
            optimizer,
            variables_to_train=trainable_variables)

      # Restore setup
      if FLAGS.trained_mentornet_dir is not None:
        ckpt_model = FLAGS.trained_mentornet_dir
        if os.path.isdir(FLAGS.trained_mentornet_dir):
          ckpt_model = tf.train.latest_checkpoint(ckpt_model)

        # fix the mentornet parameters
        variables_to_restore = slim.get_variables_to_restore(
            # TODO(lujiang): mentornet_inputs or mentor_inputs?
            include=['mentornet', 'mentornet_inputs'])
        iassign_op1, ifeed_dict1 = tf.contrib.framework.assign_from_checkpoint(
            ckpt_model, variables_to_restore)

        # Create an initial assignment function.
        def init_assign_fn(sess):
          tf.logging.info('Restore using customer initializer %s', '.' * 10)
          sess.run(iassign_op1, ifeed_dict1)
      else:
        init_assign_fn = None

      tf.logging.info('-' * 20 + 'MentorNet' + '-' * 20)
      tf.logging.info('loaded pretrained mentornet from %s', ckpt_model)
      tf.logging.info('loss_p_percentile=%3f', FLAGS.loss_p_percentile)
      tf.logging.info('burn_in_epoch=%d', FLAGS.burn_in_epoch)
      tf.logging.info('fixed_epoch_after_burn_in=%s',
                      FLAGS.fixed_epoch_after_burn_in)
      tf.logging.info('loss_moving_average_decay=%3f',
                      FLAGS.loss_moving_average_decay)
      tf.logging.info('example_dropout_rates %s', ','.join(
          str(t) for t in dropout_rates))
      tf.logging.info('-' * 20)

      saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=5)

      # Run training.
      slim.learning.train(
          train_op=train_op,
          logdir=FLAGS.train_log_dir,
          master=FLAGS.master,
          is_chief=FLAGS.task == 0,
          saver=saver,
          session_config=config,
          number_of_steps=max_step_run,
          init_fn=init_assign_fn,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs)


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_id

  if FLAGS.studentnet == 'resnet101':
    train_resnet_mentornet(FLAGS.max_number_of_steps)
  elif FLAGS.studentnet == 'inception':
    train_inception_mentornet(FLAGS.max_number_of_steps)
  else:
    tf.logging.error('unknown backbone student network %s', FLAGS.studentnet)


if __name__ == '__main__':
  tf.app.run()
