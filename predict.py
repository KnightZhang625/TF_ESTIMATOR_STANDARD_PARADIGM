# coding:utf-8

import numpy as np
import tensorflow as tf
from tensorflow.contrib import predictor
from pathlib import Path

from config import config as _cg

def predict():
  # find the pb file
  subdirs = [x for x in Path(_cg.infer_pb_path).iterdir()
              if x.is_dir() and 'temp' not in str(x)]
  latest_model = str(sorted(subdirs)[-1])
  print(latest_model)
  predict_fn = predictor.from_saved_model(latest_model)
  
  return predict_fn

if __name__ == '__main__':
  predict_fn = predict()

  data = [[1, 2]]   # although single data, however, must be batch format
  data_batch = [[2, 3], [5, 6], [7, 8]]
  features = {'input_x': np.array(data, dtype=np.float32)}
  features_batch = {'input_x': np.array(data_batch, dtype=np.float32)}

  predictions = predict_fn(features)['result_1']
  predictions_batch = predict_fn(features_batch)['result_1']

  print(predictions)
  print(predictions_batch)
