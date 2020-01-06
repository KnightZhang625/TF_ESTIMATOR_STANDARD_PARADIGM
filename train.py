# coding:utf-8
# Produced by Andysin Zhang
# Start Data: 06_Jan_2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import sys
import tensorflow as tf

from pathlib import Path
PROJECT_PATH = Path(__file__).absolute().parent
sys.path.insert(0, str(PROJECT_PATH))

from utils.setup import Setup
from utils.log import log_info as _info
from utils.log import log_error as _error
setup = Setup()

from model import Model
from config import config as _cg
from load_data import train_input_fn, server_input_fn

def model_fn_builder():
  """returns 'model_fn' closure for the Estimator."""

  def model_fn(features, labels, mode, params):
    """the above formal parameters are necessary."""
    # display the features
    _info('*** Features ***')
    for name in sorted(features.keys()):
      tf.logging.info(' name = %s, shape = %s'%(name, features[name].shape))
    
    # get the input feature
    # TODO customized define
    input_x = features['input_x']

    # define the model
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = Model(config=_cg,
                  input_x=input_x)
    output = model.get_result()
    
    # TRAIN, EVAL, PREDICT
    if mode == tf.estimator.ModeKeys.PREDICT:
      # TODO customized define
      predict_results = {'result_1': output}
      output_spec = tf.estimator.EstimatorSpec(mode, predictions=predict_results)
    else:
      if mode == tf.estimator.ModeKeys.TRAIN:
        # TODO customized define
        labels = tf.reshape(labels, [-1])
        output = tf.reshape(output, [-1])
        loss = tf.keras.losses.MSE(labels, output)
        learning_rate = tf.train.polynomial_decay(_cg.learning_rate, 
                                                  tf.train.get_or_create_global_step(),
                                                  _cg.train_steps,
                                                  end_learning_rate=_cg.end_learning_rate,
                                                  power=1.0,
                                                  cycle=False)
        optimizer = tf.train.AdamOptimizer(learning_rate, name='optimizer')
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss, tvars, colocate_gradients_with_ops=_cg.colocate_gradients_with_ops)
        clipper_gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
        train_op = optimizer.apply_gradients(zip(clipper_gradients, tvars), global_step=tf.train.get_global_step())
        
        logging_hook = tf.train.LoggingTensorHook({'loss': loss, 'lr': learning_rate}, every_n_iter=_cg.print_info_interval)
        output_spec = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
      elif mode == tf.estimator.ModeKeys.EVAL:
        # TODO
        raise NotImplementedError
    return output_spec
  
  return model_fn

def main():
  # make a directory to save the model
  Path(_cg.save_model_path).mkdir(exist_ok=True)

  model_fn = model_fn_builder()

  gpu_config = tf.ConfigProto()
  gpu_config.gpu_options.allow_growth = True

  run_config = tf.contrib.tpu.RunConfig(
    session_config=gpu_config,
    keep_checkpoint_max=_cg.keep_checkpoint_max,
    save_checkpoints_steps=_cg.save_checkpoints_steps,
    model_dir=_cg.save_model_path)
  
  estimator = tf.estimator.Estimator(model_fn, config=run_config)
  estimator.train(train_input_fn)

def package_model(ckpt_path, pb_path):
  model_fn = model_fn_builder()
  estimator = tf.estimator.Estimator(model_fn, ckpt_path)
  estimator.export_saved_model(pb_path, server_input_fn)

if __name__ == '__main__':
  if sys.argv[1] == 'train':
    main()
  elif sys.argv[1] == 'package':
    package_model(_cg.save_model_path, _cg.infer_pb_path)
  else:
    _error('Unknown Parameter: {}'.format(sys.argv[1]))
    _info('Choose from [train | package]')