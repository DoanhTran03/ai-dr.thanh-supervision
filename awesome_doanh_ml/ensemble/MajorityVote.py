import random
import pandas as pd
import numpy as np
from awesome_doanh_ml import helper
from awesome_doanh_ml.base import Algorithm
from awesome_doanh_ml.base.ensemble.EnsembleAlgorithm import EnsembleAlgorithm
from multipledispatch import dispatch

class MajorityVote (EnsembleAlgorithm):
  def __init__(self, estimators):
    super().__init__("Majority Vote", None, len(estimators))
    self.estimators = estimators

  @dispatch(pd.DataFrame)
  def fit (self, data):

    for inx, estimator in enumerate(self.estimators):
      estimator.fit(data.drop('target', axis = 1), data['target'])

    return self.estimators

  @dispatch(pd.DataFrame, pd.Series)
  def fit (self, x, y):

      data = pd.concat([x, y], axis = 1)
      return self.fit(data)


  def predict (self, data):
      predictions = []
      for estimator in self.estimators:
        proba_predictions = estimator.predict_proba(data)
        prediction = []
        for proba_prediction in proba_predictions:
            max_element = max(proba_prediction)
            max_index = proba_prediction.tolist().index(max_element)
            prediction.append(max_index)
        predictions.append(prediction)

      voted_prediction = []
      for inx in range(data.shape[0]):
          pre_values = []
          for iny, estimator in enumerate(self.estimators):
              pre_values.append(predictions[iny][inx])
          result = helper.most_frequent(pre_values)
          voted_prediction.append(result)

      return voted_prediction
      # predictions = []
      # for inx in range(data.shape[0]):
      #   sum_proba = [0,0,0]
      #   for iny, estimator in enumerate(self.estimators):
      #       sum_proba = np.add(sum_proba, proba_predictions[iny][inx])
      #
      #   max_element = max(sum_proba)
      #   max_index = sum_proba.tolist().index(max_element)
      #   predictions.append(max_index)