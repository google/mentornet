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

"""Contains code for loading and preprocessing the CIFAR data."""

import cifar100_dataset
import cifar10_dataset
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.data import dataset_data_provider

datasets_map = {
    'cifar10': cifar10_dataset,
    'cifar100': cifar100_dataset,
}


def provide_resnet_data(dataset_name,
                        split_name,
                        batch_size,
                        dataset_dir=None,
                        num_epochs=None):
  """Provides batches of CIFAR images for resnet.

  Args:
    dataset_name: Eiether 'cifar10' or 'cifar100'.
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the MNIST data can be found.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 1]
    one_hot_labels: A `Tensor` of size [batch_size, NUM_CLASSES], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of total samples in the dataset.
    num_classes: The number of total classes in the dataset.


  Raises:
    ValueError: If `split_name` is not either 'train' or 'test'.
  """
  dataset = get_dataset(dataset_name, split_name, dataset_dir=dataset_dir)

  # num_epochs = 1 if split_name == 'test' else None
  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'),
      num_epochs=num_epochs)
  if dataset_name == 'cifar100':
    [image, label] = provider.get(['image', 'fine_label'])
  else:
    [image, label] = provider.get(['image', 'label'])

  image = tf.to_float(image)

  image_size = 32
  if split_name == 'train':
    image = tf.image.resize_image_with_crop_or_pad(image, image_size + 4,
                                                   image_size + 4)
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
  else:
    image = tf.image.resize_image_with_crop_or_pad(image, image_size,
                                                   image_size)
    image = tf.image.per_image_standardization(image)

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=1,
      capacity=5 * batch_size,
      allow_smaller_final_batch=True)

  one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
  one_hot_labels = tf.squeeze(one_hot_labels, 1)
  return images, one_hot_labels, dataset.num_samples, dataset.num_classes


def provide_cifarnet_data(dataset_name,
                          split_name,
                          batch_size,
                          dataset_dir=None,
                          num_epochs=None):
  """Provides batches of CIFAR images for cifarnet.

  Args:
    dataset_name: Eiether 'cifar10' or 'cifar100'.
    split_name: Either 'train' or 'test'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the MNIST data can be found.
    num_epochs: The number of times each data source is read. If left as None,
      the data will be cycled through indefinitely.

  Returns:
    images: A `Tensor` of size [batch_size, 32, 32, 1]
    one_hot_labels: A `Tensor` of size [batch_size, NUM_CLASSES], where
      each row has a single element set to one and the rest set to zeros.
    num_samples: The number of total samples in the dataset.
    num_classes: The number of total classes in the dataset.

  Raises:
    ValueError: If `split_name` is not either 'train' or 'test'.
  """
  dataset = get_dataset(dataset_name, split_name, dataset_dir=dataset_dir)
  # num_epochs = 1 if split_name == 'test' else None
  provider = dataset_data_provider.DatasetDataProvider(
      dataset,
      common_queue_capacity=2 * batch_size,
      common_queue_min=batch_size,
      shuffle=(split_name == 'train'),
      num_epochs=num_epochs)

  if dataset_name == 'cifar100':
    [image, label] = provider.get(['image', 'fine_label'])
  else:
    [image, label] = provider.get(['image', 'label'])

  image_size = 32
  image = tf.to_float(image)

  # preprocess the images.
  if split_name == 'train':
    padding = image_size / 4
    image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
    image = tf.random_crop(image, [image_size, image_size, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)
  else:
    image = tf.image.resize_image_with_crop_or_pad(image, image_size,
                                                   image_size)
    image = tf.image.per_image_standardization(image)

  # Creates a QueueRunner for the pre-fetching operation.
  images, labels = tf.train.batch(
      [image, label],
      batch_size=batch_size,
      num_threads=1,
      capacity=5 * batch_size,
      allow_smaller_final_batch=True)

  one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
  one_hot_labels = tf.squeeze(one_hot_labels, 1)
  return images, one_hot_labels, dataset.num_samples, dataset.num_classes


def get_dataset(name, split_name, **kwargs):
  """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, name of the dataset.
    split_name: A train/test split name.
    **kwargs: Extra kwargs for get_split, for example dataset_dir.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if dataset unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  dataset = datasets_map[name].get_split(split_name, **kwargs)
  dataset.name = name
  return dataset

