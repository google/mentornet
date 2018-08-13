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

"""Reader used to learn MentorNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import numpy as np


class DataSet(object):
  """ The Dataset class"""

  def __init__(self, indir, split_name):
    self._data = pickle.load(open(os.path.join(indir, split_name + '.p'), 'rb'))
    self._num_examples = self._data.shape[0]
    self.feat_dim = self._data.shape[1] - 1
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def feature_dim(self):
    return self.feat_dim

  @property
  def is_binary_label(self):
    unique_labels = np.unique(self._data[:, -1])
    if len(unique_labels) == 2 and (0 in unique_labels) and (
        1 in unique_labels):
      return True
    return False

  def next_batch(self, batch_size):
    """Store everything in memory for small-scale application."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # finished epoch
      self._epochs_completed += 1
      # shuffle the data
      perm = np.arange(self.num_examples)
      np.random.shuffle(perm)

      self._data = self._data[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size

    end = self._index_in_epoch

    cur_data = self._data[start:end]
    return cur_data
