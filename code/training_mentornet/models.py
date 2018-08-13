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

"""Defines the various functions for training the pre-defined MentorNet."""
import numpy as np
import scipy.stats as st

import tensorflow as tf


def logistic(inputs):
  """A baseline logistic model."""
  feat_dim = int(inputs.get_shape()[1])
  with tf.variable_scope('logistic'):
    layer_1 = tf.add(
        tf.matmul(inputs, tf.get_variable('W1', [feat_dim, 1])),
        tf.get_variable('b1', [1]))
    return layer_1


def mlp(inputs, num_hidden_nodes=10):
  """A baseline MLP model."""
  feat_dim = int(inputs.get_shape()[1])
  with tf.variable_scope('mlp'):
    layer_1 = tf.add(
        tf.matmul(inputs,
                  tf.Variable(tf.random_normal([feat_dim, num_hidden_nodes]))),
        tf.Variable(tf.random_normal([num_hidden_nodes])))
    layer_1 = tf.nn.tanh(layer_1)
    # output layer has the linear activation
    out_layer = tf.matmul(
        layer_1,
        tf.Variable(tf.random_normal([num_hidden_nodes, 1])) + tf.Variable(
            tf.random_normal([1])))
    return out_layer


def vstar_baseline(inbatch, **kwargs):
  """Variable star function for equally weighting every sample.

  Args:
    inbatch: a numpy array with the following
      Index 0: Loss
      Index 1: Loss difference from moving average
      Index 3: Label
      Index 4: Epoch
   **kwargs: hyper-parameter specified in vtsar_gamma

  Returns:
    v: [batch_size, 1] weight vector.
  """
  del kwargs  # Unused.
  v = np.ones(inbatch.shape[0])
  return v


def vstar_self_paced(inbatch, **kwargs):
  """Variable star function for self-paced learning.

  Args:
    inbatch: a numpy array with the following
      Index 0: Loss
      Index 1: Loss difference from moving average
      Index 3: Label
      Index 4: Epoch
   **kwargs: hyper-parameter specified in vtsar_gamma

  Returns:
    v: [batch_size, 1] weight vector.
  """
  del kwargs  # Unused.
  loss_diff = inbatch[:, 1]
  v = np.copy(loss_diff)
  v[np.where(loss_diff <= 0)] = 1
  v[np.where(loss_diff > 0)] = 0
  return v


def vstar_hard_example_mining(inbatch, **kwargs):
  """Variable v_star function for self-paced learning.

  Args:
    inbatch: a numpy array with the following
      Index 0: Loss
      Index 1: Loss difference from moving average
      Index 3: Label
      Index 4: Epoch
    **kwargs: hyper-parameter specified in vtsar_gamma

  Returns:
    v: [batch_size, 1] weight vector.
  """
  del kwargs  # Unused.
  loss_diff = inbatch[:, 1]
  y = inbatch[:, 2]
  v = np.copy(loss_diff)
  v[np.where(loss_diff >= 0)] = 1
  v[np.where(loss_diff < 0)] = 0
  v[np.where(y > 0)] = 1  # Select all positive
  return v


def vstar_focal_loss(inbatch, **kwargs):
  """Variable v_star function for focal loss.

  Args:
    inbatch: a numpy array with the following
      Index 0: Loss
      Index 1: Loss difference from moving average
      Index 3: Label
      Index 4: Epoch
    **kwargs: hyper-parameter specified in vtsar_gamma

  Returns:
    v: [batch_size, 1] weight vector.
  """
  if 'vstar_gamma' in kwargs:
    gamma = kwargs['vstar_gamma']
  else:
    gamma = 2
  assert gamma > 0

  loss = inbatch[:, 0]
  v = np.power((1 - np.exp(-1 * loss)), gamma)
  return v


def vstar_spcl_linear(inbatch, **kwargs):
  """Variable v_star function for self-paced curriculum learning (linear).

  Args:
    inbatch: a numpy array with the following
      Index 0: Loss
      Index 1: Loss difference from moving average
      Index 3: Label
      Index 4: Epoch
    **kwargs: hyper-parameter specified in vtsar_gamma

  Returns:
    v: [batch_size, 1] weight vector.
  """
  if 'vstar_gamma' in kwargs:
    gamma = kwargs['vstar_gamma']
  else: gamma = 1

  assert gamma != 0

  loss_diff = inbatch[:, 1]
  v = -1.0 / gamma * loss_diff + 1
  v = np.maximum(np.minimum(v, 1), 0)
  return v


def vstar_mentornet_pd(inbatch, **kwargs):
  """Variable v_star function for the pre-defined mentornet: MentorNet PD.

  Args:
    inbatch: a numpy array with the following
      Index 0: Loss
      Index 1: Loss difference from moving average
      Index 3: Label
      Index 4: Epoch
    **kwargs: hyper-parameter specified in vtsar_gamma

  Returns:
    v: [batch_size, 1] weight vector.
  """
  epoch = inbatch[:, 3]

  v1 = vstar_self_paced(inbatch, **kwargs)
  v2 = vstar_spcl_linear(inbatch, **kwargs)

  v = np.zeros(len(epoch))

  ids = np.where(epoch >= 90)
  v[ids] = v2[ids]

  ids = np.where(epoch < 90)
  v[ids] = v1[ids]

  ids = np.where(epoch <= 18)
  v[ids] = 1
  return v


def mean_confidence_interval(data, confidence=0.9):
  """Calculation of mean confidence interval.

  Args:
    data: the data for which the confidence interval has to be found
    confidence: the confidence threshold

  Returns:
    mean of data, mean confidence interval
  """
  mean, sem, m = np.mean(data), st.sem(data), st.t.ppf((1 + confidence) / 2.,
                                                       len(data) - 1)
  return mean, m * sem
