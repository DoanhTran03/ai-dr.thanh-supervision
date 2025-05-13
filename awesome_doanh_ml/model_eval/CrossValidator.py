import random
import math
import pandas as pd
import statistics
from sklearn.metrics import accuracy_score, f1_score
import xlsxwriter
import matplotlib.pyplot as pl
from collections import deque

class TrainTestPair:
  def __init__(self, train_df, test_df):
    self.train_df = train_df
    self.test_df = test_df

class ModelPredictions:
    def __init__(self, model, predictions):
        self.model = model
        self.predictions = predictions

class ModelMetrics:
    def __init__(self, model, acc_scores, f1_scores, acc_mean, f1_mean, acc_dev, f1_dev):
        self.model = model
        self.acc_scores = acc_scores
        self.f1_scores = f1_scores
        self.acc_mean = acc_mean
        self.f1_mean = f1_mean
        self.acc_dev = acc_dev
        self.f1_dev = f1_dev

class TruthModelColors:
    def __init__(self,truth_val_color="green", modelcolor_s=["blue","red"]):
        self.truth_val_color = truth_val_color
        self.modelcolor_s = deque(modelcolor_s)

    def get_truth_val_color(self):
        return self.truth_val_color

    def get_model_color(self):
        return self.modelcolor_s.popleft()

    def append_model_color(self, color):
        self.modelcolor_s.append(color)

class CrossValidator:

    #Parameter:
    # (DataFrame) dataframe - contains dataset
    # (int) number_of_folds - number of division on the dataset
    def __init__(self, models, dataset ,fold_number):
        self.models = models
        self.dataset = dataset
        self.fold_number = fold_number
        self.folds = self.__slide_df_to_n_folds()
        self.train_test_pairs = self.__prepare_train_test_pairs()
        self.truth_val_tests = self.__prepare_truth_val_tests()
        self.model_predictions_s = self.__cross_validate()
        self.model_metrics_s = self.__prepare_model_metrics_s()

    #Return: List of Dataframes that are divided from the original Dataframe

    def __slide_df_to_n_folds (self) :
        #--General function variables
        wrapper_array = []
        list_of_divided_dataframe = []

        #--Shuffle and divide the index of data_frame
        index_of_df = list(range(self.dataset.shape[0]))
        random.shuffle(index_of_df)
        #Calculate the number of fold will be divided into
        number_of_element_per_fold = math.ceil(len(index_of_df)/self.fold_number)
        #Divide the list of array into the corresponding fold
        for k in range (0, len(index_of_df), number_of_element_per_fold):
            wrapper_array.append(index_of_df[k:k+number_of_element_per_fold])

        #--Divide the dataframe based on the index was created
        for m in range(len(wrapper_array)):
            new_dataframe = self.dataset.iloc[wrapper_array[m]]
            list_of_divided_dataframe.append(new_dataframe)

        return list_of_divided_dataframe

    #Parameter:
    # (List) list of dataframes

    #Process: for each train-test pair created, test fold will be picked sequentially from the input list while the train fold will be the remaining folds
    #Ex: Input: x,y,z -> (x,(y,z)), ((x,z),y), ((x,y),z).

    #Return: list of TrainTestPair

    def __prepare_train_test_pairs (self):
        train_test_pairs = []
        for inx, fold in enumerate(self.folds):
            train_dfs = self.folds[:inx] + self.folds[inx+1:]
            train_df = pd.concat(train_dfs,axis=0)
            test_df = self.folds[inx]

            train_test_pairs.append(TrainTestPair(train_df, test_df))

        return train_test_pairs


    #Parameters:
    # (List)<TrainTestPair> - a list of train test pair
    #Return: List of dataframe of label of test folds

    def __prepare_truth_val_tests (self):
        truth_val_tests = []

        for train_test_pair in self.train_test_pairs:
            truth_val_tests.append(train_test_pair.test_df["target"])

        return truth_val_tests

    #Parameters:
    # (List) - list of model
    # (List) - list of train test pair
    #Process: For each model, we train in each train fold and get predictions for test fold
    #Return: (List) - list of ModelPredictions

    def __cross_validate(self):
        model_predictions_s = []

        for model in self.models:
            predictions = []

            for train_test_pair in self.train_test_pairs:
                train_data = train_test_pair.train_df
                test_data = train_test_pair.test_df

                #--Train model on training fold
                model.fit(train_data.drop("target", axis=1), train_data["target"])
                #--Make prediction
                prediction_result = model.predict(test_data.drop(columns='target'))

                predictions.append(prediction_result)

            model_predictions_s.append(ModelPredictions(model,predictions))

        return model_predictions_s

    # Parameters:
    # (List) <ModelMetrics> - list of model and their evaluation metrics

    # Return: None - evaluation metrics for each model will be saved into Excel

    def save_evalMetrics_to_excel(self):
        workbook = xlsxwriter.Workbook("Result.xlsx")

        for inx, model_metrics in enumerate(self.model_metrics_s):

            worksheet_fold = workbook.add_worksheet(model_metrics.model.name)

            row = 1

            worksheet_fold.write(0, 0, f"Fold number")
            worksheet_fold.write(0, 1, f"Accuracy score")

            for iny, mse in enumerate(model_metrics.acc_scores):
                worksheet_fold.write(row, 0, f"{iny}")
                worksheet_fold.write(row, 1, f"{mse}")
                row += 1

            worksheet_fold.write(row, 0, f"Average value: {model_metrics.acc_mean}")
            row += 1
            worksheet_fold.write(row, 0, f"Standard Derivation: {model_metrics.acc_dev}")
            row += 1

            row += 3

            worksheet_fold.write(row, 0, f"Fold number")
            worksheet_fold.write(row, 1, f"F1 Score")

            row += 1

            for iny, mse in enumerate(model_metrics.f1_scores):
                worksheet_fold.write(row, 0, f"{iny}")
                worksheet_fold.write(row, 1, f"{mse}")
                row += 1

            worksheet_fold.write(row, 0, f"Average value: {model_metrics.f1_mean}")
            row += 1
            worksheet_fold.write(row, 0, f"Standard Derivation: {model_metrics.f1_dev}")
            row += 1

            row += 3

        workbook.close()

        return

    #Parameters:
    # (List) <ModelPredictions> - list of model and their prediction for each test fold
    # (List) - list of test truth values according to each test fold

    #Return:
    # (List) <ModelMetrics> - list of model and their evaluation metrics

    def __prepare_model_metrics_s(self):
        model_metrics_s = []

        for inx, model_predictions in enumerate(self.model_predictions_s):

            acc_scores = []
            f1_scores = []

            for iny, truth_val_test in enumerate(self.truth_val_tests):
                acc_sc = accuracy_score(truth_val_test, model_predictions.predictions[iny])
                f1_sc = f1_score(truth_val_test, model_predictions.predictions[iny], average='macro')

                acc_scores.append(acc_sc)
                f1_scores.append(f1_sc)

            acc_mean = statistics.mean(acc_scores)
            f1_mean = statistics.mean(f1_scores)

            acc_dev = statistics.stdev(acc_scores)
            f1_dev = statistics.stdev(f1_scores)

            model_metrics_s.append(ModelMetrics(model_predictions.model, acc_scores, f1_scores, acc_mean, f1_mean, acc_dev, f1_dev))

        return model_metrics_s

    #Parameters:
    # (TruthModelColors) - object defines the color of truth value color and the prediciton of each model
    #Output:
    # Graphs demonstrate the number of test fold and the prediction from each model

    def plot_cv_results(self, truth_modelcolors):
        #Check number of model and number of color according to
        if(len(truth_modelcolors.modelcolor_s) != len(self.models)):
            raise Exception("Number of model colors does not match number of models")

        # Configure graph properties
        pl.rcParams['figure.figsize'] = [20, 15]
        pl.rcParams["figure.autolayout"] = True

        # Define color line for different model
        truth_modelcolors = TruthModelColors()

        fig, ax = pl.subplots(len(self.truth_val_tests))

        #Fo each of test fold, we compare the predictions of models
        for inx, truth_val_test in enumerate(self.truth_val_tests):
            ax[inx].plot(truth_val_test.tolist(), truth_modelcolors.get_truth_val_color(), label="Truth value")
            ax[inx].set_title(f"Fold number {inx}")
            ax[inx].legend(loc="upper right")

            for model_predictions in self.model_predictions_s:
                model_name = model_predictions.model.name
                modelcolor = truth_modelcolors.get_model_color()
                truth_modelcolors.append_model_color(modelcolor)
                ax[inx].plot(model_predictions.predictions[inx], linestyle="dotted", color=f"{modelcolor}", label=model_name)
                ax[inx].legend(loc="upper right")

        fig.tight_layout()