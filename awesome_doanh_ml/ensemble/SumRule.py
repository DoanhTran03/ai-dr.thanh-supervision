import random
import pandas as pd
import numpy as np
from awesome_doanh_ml import helper
from awesome_doanh_ml.base import Algorithm
from awesome_doanh_ml.base.ensemble.EnsembleAlgorithm import EnsembleAlgorithm
from multipledispatch import dispatch

class SumRule (EnsembleAlgorithm):
  def __init__(self, estimators):
    super().__init__("Sum Rule", None, len(estimators))
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
      proba_predictions = []
      for estimator in self.estimators:
        prediction = estimator.predict_proba(data)
        proba_predictions.append(prediction)

      predictions = []
      for inx in range(data.shape[0]):
        sum_proba = [0,0,0]
        for iny, estimator in enumerate(self.estimators):
            sum_proba = np.add(sum_proba, proba_predictions[iny][inx])

        max_element = max(sum_proba)
        max_index = sum_proba.tolist().index(max_element)
        predictions.append(max_index)

      return predictions