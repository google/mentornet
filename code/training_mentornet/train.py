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

"""Trains MentorNet. After trained, we update the MentorNet in the Algorithm.

Run the following command before running the python code.
export PYTHONPATH="$PYTHONPATH:$PWD/code/"
"""
import os
import time
import numpy as np
import reader
import tensorflow as tf
import utils

flags = tf.app.flags
flags.DEFINE_string('train_dir', '', 'Training output directory')
flags.DEFINE_string('data_path', '', 'data_path')
flags.DEFINE_integer('mini_batch_size', 32, 'training mini batch size')
flags.DEFINE_float('max_step_train', 3e4, 'max steps to train')
flags.DEFINE_float('learning_rate', 0.1, 'starting learning rate')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def create_train_op(loss,
                    global_step,
                    starter_learning_rate,
                    decay_steps=500,
                    max_grad_norm=5):
  """Creates an adam optimizer with the given learning rate.

  Args:
    loss: the loss from the model
    global_step: number of steps model has taken
    starter_learning_rate: learning rate to start at
    decay_steps: step size for decay
    max_grad_norm: the cap of the gradient norm

  Returns:
    the optimizer function for training
  """
  with tf.name_scope('train'):
    # Create the gradient descent optimizer with the given learning rate.
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate, global_step, decay_steps, 0.9, staircase=True)
    tf.add_to_collection('lr', learning_rate)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Tune the optimizer GradientDescentOptimizer AdadeltaOptimizer
    with tf.control_dependencies([loss]):
      train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=global_step)
    return train_op


def _eval_ts_once(test_data, session, loss, v, input_data_pl, v_truth_pl,
                  mini_batch_size):
  """Evaluates the model once."""
  batch = test_data.next_batch(mini_batch_size)
  epoch_size = int(np.floor(test_data.num_examples / mini_batch_size))
  accumulated_loss = 0.0

  for _ in xrange(epoch_size):
    batch = test_data.next_batch(mini_batch_size)
    v_truth = batch[:, 4]
    v_truth = np.reshape(v_truth, (mini_batch_size, 1))

    feed_dict = {input_data_pl: batch[:, 0:4], v_truth_pl: v_truth}

    loss_val, _ = session.run([loss, v], feed_dict=feed_dict)
    accumulated_loss += loss_val

  return accumulated_loss / epoch_size


def train_mentornet():
  """Runs the model on the given data."""
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)

  tf.logging.info('Start loading the data')
  train_data = reader.DataSet(FLAGS.data_path, 'tr')
  test_data = reader.DataSet(FLAGS.data_path, 'ts')
  tf.logging.info('Finish loading the data')

  mini_batch_size = FLAGS.mini_batch_size
  epoch_size = int(np.floor(train_data.num_examples / FLAGS.mini_batch_size))

  parameter_info = ['Hyper Parameter Info:']
  parameter_info.append('=' * 20)
  parameter_info.append('data_dir={}'.format(FLAGS.data_path))
  parameter_info.append('#train_examples={}'.format(train_data.num_examples))
  parameter_info.append('#test_examples={}'.format(test_data.num_examples))
  parameter_info.append('is_binary_label={}'.format(train_data.is_binary_label))
  parameter_info.append(
      'mini_batch_size = {}\nstarter_learning_rate = {}'.format(
          mini_batch_size, FLAGS.learning_rate))
  parameter_info.append('#iterations per epoch = {}'.format(epoch_size))
  parameter_info.append('=' * 20)
  tf.logging.info('\n'.join(parameter_info))

  tf.logging.info('Start constructing computation graph.')
  tf.reset_default_graph()

  with tf.device('/device:GPU:0'):
    # global step counter
    global_step = tf.train.create_global_step()

    input_data_pl = tf.placeholder(
        tf.float32,
        shape=(mini_batch_size, train_data.feature_dim),
        name='input_data_pl')

    v_truth_pl = tf.placeholder(
        tf.float32, shape=(mini_batch_size, 1), name='v_truth_pl')
    # pylint: disable=undefined-variable
    v = utils.mentornet_nn(input_data_pl)

    if train_data.is_binary_label:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=v_truth_pl, logits=v)
      loss = tf.reduce_mean(loss)
    else:
      v = tf.nn.sigmoid(v)
      loss = tf.nn.l2_loss(v_truth_pl - v) * 2.0 / mini_batch_size

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = create_train_op(
        loss, global_step, FLAGS.learning_rate, decay_steps=1000)
    lr = tf.get_collection('lr')[0]  # learning rate

    # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()
  with tf.Session() as session:
    # you need to clear the training directory
    init = tf.global_variables_initializer()
    session.run(init)

    tf.logging.info('Start Training')

    session.run(init)
    accumulated_loss = 0.0
    accumulated_time = 0.0
    epoch_num = 0

    mse = []
    for _ in xrange(int(FLAGS.max_step_train)):
      start_time = time.time()
      # Get the next mini-batch data
      batch = train_data.next_batch(mini_batch_size)

      # batch_epoch = batch[:,3]
      v_truth = batch[:, 4]
      v_truth = np.reshape(v_truth, (mini_batch_size, 1))

      feed_dict = {input_data_pl: batch[:, 0:4], v_truth_pl: v_truth}

      # Run the graph
      global_step_val, _, loss_val, _, lr_val = session.run(
          [global_step, train_op, loss, v, lr], feed_dict=feed_dict)

      duration = time.time() - start_time

      accumulated_loss = accumulated_loss + loss_val
      accumulated_time = accumulated_time + duration

      if global_step_val % epoch_size == 0:
        # reach an epoch
        epoch_num = epoch_num + 1
        print 'epoch={:04d} global_step={:04d} lr={:.4f} time={:.2f}'.format(
            epoch_num, global_step_val, lr_val, accumulated_time)
        train_avg_loss = accumulated_loss / epoch_size
        test_avg_loss = _eval_ts_once(test_data, session, loss, v,
                                      input_data_pl, v_truth_pl,
                                      mini_batch_size)
        print 'tr_loss={:.6f}, ts_loss={:.6f}'.format(train_avg_loss,
                                                      test_avg_loss)

        mse.append(test_avg_loss)
        accumulated_loss = 0.0
        accumulated_time = 0.0
        # save model
        saver.save(
            session,
            os.path.join(FLAGS.train_dir, '{}.model'.format('mentornet')),
            global_step=global_step_val)
    for i in range(len(mse)):
      print 'Epoch {}: MSE = {:.4f}'.format(i, mse[i])
    saver.save(
        session,
        os.path.join(FLAGS.train_dir, '{}.model'.format('mentornet')),
        global_step=global_step_val)
    return mse


def eval_model_once(data_path, model_dir):
  """Evaluates the model.

  Args:
    data_path: path where the data is stored.
    model_dir: path where the model checkpoints are stored.

  Returns:
    average loss
  """
  tf.reset_default_graph()
  mini_batch_size = 32
  test_data = reader.Dataset(data_path, 'ts')
  input_data_pl = tf.placeholder(
      tf.float32,
      shape=(mini_batch_size, test_data.feature_dim),
      name='input_data_pl')

  v_truth_pl = tf.placeholder(
      tf.float32, shape=(mini_batch_size, 1), name='v_truth_pl')
  # pylint: disable=undefined-variable
  v = utils.mentornet_nn(input_data_pl)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=v_truth_pl, logits=v)
  loss = tf.reduce_mean(loss)

  saver = tf.train.Saver()
  np.random.seed(0)
  with tf.Session() as session:
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(session, ckpt.model_checkpoint_path)
    return _eval_ts_once(test_data, session, loss, v, input_data_pl, v_truth_pl,
                         mini_batch_size)


def main(_):
  if not os.path.exists(FLAGS.train_dir):
    os.makedirs(FLAGS.train_dir)
  train_mentornet()


if __name__ == '__main__':
  tf.app.run()
