# Description: A function to create box plot graph to compare performance of RandomSubspace on different number of estimator

# Input:
# (Ensemble Algorithm) An Ensemble model
# (Algorithm)
# (List) - a list of number of estimators input for the ensemble learning algorithm

# Return: - a box plot graph comparing a model evaluation with different estimators

import matplotlib.pyplot as pl
from awesome_doanh_ml.model_eval.CrossValidator import CrossValidator


def box_plt (model, algorithm, estimator_nums, data, fold_num = 5):
    #List of random subspace classifier
    rs_c_s  = []

    #Add all Random subspace algorithm with different number of estimator to a list
    for estimator_num in estimator_nums:
        new_model = model(algorithm, estimator_num)
        rs_c_s.append(new_model)

    cr_validator = CrossValidator(rs_c_s, data, fold_num)

    acc_scores_s = []
    f1_scores_s = []

    for model_metrics in cr_validator.model_metrics_s:
        acc_scores_s.append(model_metrics.acc_scores)
        f1_scores_s.append(model_metrics.f1_scores)

    fig = pl.figure(figsize =(10, 7))
    fig.suptitle('Accuracy Score', fontsize=16)
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(acc_scores_s)
    ax.set_xticklabels(estimator_nums)

    fig2 = pl.figure(figsize =(10, 7))
    fig2.suptitle('F1 Score', fontsize=16)
    ax2 = fig2.add_axes([0, 0, 1, 1])
    bp = ax2.boxplot(f1_scores_s)
    ax2.set_xticklabels(estimator_nums)

    return