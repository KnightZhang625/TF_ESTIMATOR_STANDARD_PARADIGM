# coding:utf-8

import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()

from config import config as _cg

def train_generator():
  # TODO customized define
  # mock data with distribution of wx + b
  input_x = np.random.rand(100, 2)
  weights = np.ones((2, 1)) * 2
  bias = np.random.rand(100, 1)
  labels = np.dot(input_x, weights) + bias

  for idx in range(input_x.shape[0]):
    features = {'input_x': input_x[idx, ]}
    yield (features, labels[idx, ])

def train_input_fn():
  # TODO customized define
  output_types = {'input_x': tf.float32}

  # supposed 'input' should be (batch_size, feature_size)
  # the shape should be identical to the feature yield by the above function
  output_shapes = {'input_x': [None]}  
  
  dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_types=(output_types, tf.float32),
    output_shapes=(output_shapes, [None]))
  
  dataset = dataset.shuffle(_cg.buffer_size).batch(_cg.batch_size).repeat(_cg.train_steps)

  return dataset

def server_input_fn():
  """for inference"""
  # TODO customized define

  # shape has two dimensions just for batch inference
  input_x = tf.placeholder(tf.float32, shape=[None, None], name='input_x')

  # for inference input
  receiever_tensors = {'input_x': input_x}
  # for 'model_fn' read the feature
  features = {'input_x': input_x}

  return tf.estimator.export.ServingInputReceiver(features, receiever_tensors)

if __name__ == '__main__':
  for data in train_input_fn():
    print(data)
    input()