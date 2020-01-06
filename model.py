# coding:utf-8

import copy
import tensorflow as tf

# TODO customized define
class Model(object):
  def __init__(self,
               config,
               input_x):
    self.config = copy.deepcopy(config) 
    self.result = self.build_graph(input_x)
  
  def build_graph(self, x):
    with tf.variable_scope('multiple'):
      weights = tf.get_variable(name='weights', shape=[2, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
      bias = tf.get_variable(name='bias', shape=[1], initializer=tf.truncated_normal_initializer(stddev=0.01))
      y = tf.matmul(x, weights) + bias
    return y 

  def get_result(self):
    return self.result