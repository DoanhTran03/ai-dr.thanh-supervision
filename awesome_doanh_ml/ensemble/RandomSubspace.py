import random
import pandas as pd
from awesome_doanh_ml import helper
from awesome_doanh_ml.base import Algorithm
from awesome_doanh_ml.base.ensemble.EnsembleAlgorithm import EnsembleAlgorithm
from multipledispatch import dispatch

class RandomSubspace(EnsembleAlgorithm):
  def __init__(self, algorithm, n_estimators):
    super().__init__("Random Subspace", algorithm, n_estimators)

  @dispatch(pd.DataFrame)
  def fit (self, data):

    for inx in range(self.n_estimators):
      ran_feature_num = random.randint(1, len(data.columns) - 1)
      available_col_index = list(range(len(data.columns) - 1))

      #Index the available columns in the dataframe ("target" column is not included)
      choosen_col_index = random.sample(available_col_index, k = ran_feature_num)

      #Keep the columns according to the random index selected
      sub_data = data[data.columns[choosen_col_index]]
      sub_data = pd.concat([sub_data, data.target], axis = 1)

      #Initialize estimator and train on the choosen columns
      estimator = self.algorithm()
      estimator.fit(sub_data.drop('target', axis = 1), sub_data['target'])
      self.estimators.append(estimator)

    return self.estimators

  @dispatch(pd.DataFrame, pd.Series)
  def fit (self, x, y):

      # for inx in range(self.n_estimators):
      #
      #     ran_feature_num = random.randint(1, len(x.columns) - 1)
      #     available_col_index = list(range(len(x.columns) - 1))
      #
      #     #Index the available columns in the dataframe ("target" column is not included)
      #     choosen_col_index = random.sample(available_col_index, k=ran_feature_num)
      #
      #     #Keep the columns according to the random index selected
      #     sub_data = x[x.columns[choosen_col_index]]
      #
      #     #Initialize estimator and fit
      #     estimator = self.algorithm()
      #     estimator.fit(sub_data, y)
      #
      # self.estimators.append(estimator)
      # return self.estimators

      data = pd.concat([x, y], axis = 1)
      return self.fit(data)


  def predict (self, data):
      predictions = []
      for estimator in self.estimators:
        test_data = data[estimator.feature_names_in_]
        prediction = estimator.predict(test_data)
        predictions.append(prediction)

      voted_prediction = []
      for inx in range(data.shape[0]):
        pre_values = []
        for iny, estimator in enumerate(self.estimators):
            pre_values.append(predictions[iny][inx])
        result = helper.most_frequent(pre_values)
        voted_prediction.append(result)

      return voted_prediction