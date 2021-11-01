import numpy as np
import os
import matplotlib.pyplot as plt
import zipfile
import collections
from functools import partial
import csv
import implementations
from implementations import *

path_data = 'dataset/train.csv'

dataset = Dataset().from_path(path_data)

def replace_wrong_data(dataset):
  wrong_samples = dataset.x<=-999
  dataset.x[wrong_samples] = np.median(dataset.x[~wrong_samples], axis=0)

def linear_model(w, tx, y, threshold=0):
  return predict_label(tx@w, threshold), compute_rmse(y,tx,w)

def preprocess(dataset):
  return augment_with(augmentations[:13])(dataset)


replace_wrong_data(dataset)

np.random.seed(2021)
train, test = dataset.split(0.8)
train.x.shape, test.x.shape



min_features = np.min(dataset.x, axis=0)
###the log center features in 1 so to not have negative values 
log_exp = partial(log_expand, min_features=min_features) 

augmentations = [sin_expand, cos_expand, log_exp]+[partial(ploy_expand,i) for i in range(2,16)]
w, _, _, _, _ = eval_model(train, 
            test, 
            linear_model,
            augment_with(augmentations[:13]),
            opti_function = partial(ridge_regression, lambda_=4.64e-04),
            print_result=True)


ids = load_ids('dataset/test.csv')
prediction_dataset = Dataset().from_path('dataset/test.csv')
prediction_dataset.x.shape

replace_wrong_data(prediction_dataset)



augmented_prediction_dataset = preprocess(prediction_dataset)
predictions, _ = linear_model(w,augmented_prediction_dataset.x, augmented_prediction_dataset.y)

create_csv_submission(ids, predictions, 'predictions.csv')