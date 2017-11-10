import json
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr, zscore
import seaborn as sns

from collections import Counter
from sklearn import linear_model, ensemble, svm
from sklearn.preprocessing import Imputer, StandardScaler, RobustScaler, Normalizer, MinMaxScaler

TRAIN_NUM = 400
VALID_NUM = 100


def get_correlations(x, y, corr_score_dict):
    corr_dict = {}
    for name, corr in corr_score_dict.items():
        if "Kappa" not in name:
            corr_dict[name] = corr(x, y)[0]
    return corr_dict


def master_analyzer():

    with open("/home/waseem/Metrics Analysis/author_based", "rb") as f:
        (author1_metric, author2_metric, final_actions, final_objects,
         final_score) = pickle.load(f)

    with open("/home/waseem/Metrics Analysis/metric_values", "r") as f:
        output_dict = json.load(f)

    correlation_metrics = {"Pearson": pearsonr, "Kendall": kendalltau,
                           "Spearman": spearmanr}

    # bins = split_into_bins(final_score, 5)
    #
    # corr_bin_dict = {}
    #
    # for bin_name, bin in bins.items():
    #     gt = [final_score[i] for i in bin]
    #     for key in output_dict:
    #         scores = [output_dict[key][i] for i in bin]
    #         corr = get_correlations(scores, gt, correlation_metrics)
    #         # scatter_plot(gt, scores)
    #         print("Correlations are {}".format(corr))
    #         if bin_name not in corr_bin_dict:
    #             corr_bin_dict[bin_name] = {key: corr}
    #         else:
    #             corr_bin_dict[bin_name][key] = corr
    # corr = get_correlations(output_dict["METEOR"], final_score, correlation_metrics)
    # print("Correlations are {}".format(corr))
    #

    best_metrics = {}
    best_metrics_name = {}

    # _, output_dict = prepare_dataset(output_dict, True, True)

    # for key in output_dict:
    #     bins = split_into_bins(output_dict[key], 1)
    #
    #     corr_bin_dict = {}
    #
    #     for bin_name, bin in bins.items():
    #         gt = [output_dict[key][i] for i in bin]
    #         scores = [final_score[i] for i in bin]
    #         corr = get_correlations(scores, gt, correlation_metrics)
    #         # scatter_plot(gt, scores)
    #         # print("Correlations are {}".format(corr))
    #         if bin_name == "bin_5":
    #             stop = 1
    #         if not math.isnan(corr["Spearman"]):
    #             if bin_name in best_metrics:
    #                 if best_metrics[bin_name]["Spearman"] < corr["Spearman"]:
    #                     best_metrics[bin_name] = corr
    #                     best_metrics_name[bin_name] = key
    #             else:
    #                 best_metrics[bin_name] = corr
    #                 best_metrics_name[bin_name] = key
    # corr = get_correlations(output_dict["METEOR"], final_score, correlation_metrics)
    # print("Correlations are {}".format(corr))

    #
    #
    # df = pd.DataFrame(data=corr_bin_dict)
    # df.to_csv("/home/waseem/Metrics Analysis/separated.csv")


    # for key in output_dict:
    #     scatter_plot(final_objects, output_dict[key], "Human", key)

    # scatter_plot(output_dict["caption_accuracy"], output_dict["METEOR"])

    # scatter_plot(final_score, output_dict["METEOR"], "Human", "METEOR")

    # scatter_plot(author1_metric, author2_metric, "Author1", "Author2")

    # study_more(final_score, output_dict, correlation_metrics)

    # corr_dict = get_correlations(np.add(output_dict["METEOR"],
    #                                     1 * np.array(output_dict["SPICE"])),
    #                              final_score, correlation_metrics)

    # corr_dict = get_correlations(output_dict["METEOR"],
    #                              final_score, correlation_metrics)

    # print("The correlation is: {}".format(corr_dict))

    # scatter_plot(final_objects, final_score)


def scatter_plot(x, y, xlabel="x-axis", ylabel="y-axis"):
    agg_all = Counter(zip(x, y))
    z = []
    for x_, y_ in zip(x, y):
        z.append(agg_all[x_, y_])

    plt.scatter(x, y, s=z, alpha=0.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def split_into_bins(final_score, k):
    bins = {}

    min_val = np.min(final_score)

    diff = (np.max(final_score) - min_val) / k

    for i, score in enumerate(final_score):
        for j in range(k):
            if score <= min_val + (j + 1) * diff:
                if "bin_" + str(j + 1) in bins:
                    bins["bin_" + str(j + 1)].append(i)
                else:
                    bins["bin_" + str(j + 1)] = [i]
                break
    return bins


def prepare_dataset(output_dict, impute=False, scale=False):
    imputer = Imputer(missing_values=-np.inf, strategy="most_frequent", axis=0)
    scaler = StandardScaler()
    scaler = MinMaxScaler()
    # scaler = RobustScaler(quantile_range=(1.0, 99.0))
    # scaler = Normalizer()

    max_range = len(list(output_dict.values())[0]) # No of measurements
    max_rows = len(output_dict.values()) # No of metrics

    # print("output_dict keys1: {}".format(output_dict.keys()))
    # dataset = np.zeros([max_rows, max_range])

    dataset = np.zeros([max_range, max_rows])

    for j, val in enumerate(output_dict.values()):
        # if key not in ["BLEU@4", "caption_accuracy", "first_accuracy", "accuracy"]:
        dataset[:, j] = val

    if impute:
        dataset = imputer.fit_transform(dataset)

    if scale:
        dataset = scaler.fit_transform(dataset)

    new_dict = {}

    # print("output_dict keys2: {}".format(output_dict.keys()))

    for i, key in enumerate(output_dict.keys()):
        new_dict[key] = dataset[:, i]

    return dataset, new_dict


def split_into_sets(final_score, output_dict):
    dataset, _ = prepare_dataset(output_dict, True, True)

    training_set = (dataset[:TRAIN_NUM], final_score[:TRAIN_NUM])
    validation_set = (dataset[TRAIN_NUM: TRAIN_NUM + VALID_NUM],
                      final_score[TRAIN_NUM: TRAIN_NUM + VALID_NUM])
    test_set = (dataset[TRAIN_NUM + VALID_NUM:],
                final_score[TRAIN_NUM + VALID_NUM:])
    return training_set, validation_set, test_set


def study_more(final_score, output_dict, correlation_metrics):
    # output_dict = {"METEOR": output_dict["METEOR"]}
    training_set, validation_set, test_set = split_into_sets(final_score,
                                                             output_dict)
    # model = linear_model.LinearRegression()
    # model = linear_model.Lasso(alpha=0.000001)
    model = ensemble.GradientBoostingRegressor()
    # model = svm.LinearSVR()
    model.fit(*training_set)
    predictions = model.predict(validation_set[0])
    corr = get_correlations(predictions, validation_set[1], correlation_metrics)
    print("The fitted correlations are {}".format(corr))

    predictions = model.predict(test_set[0])
    corr = get_correlations(predictions, test_set[1], correlation_metrics)
    print("The test correlations are {}".format(corr))

    print("Output dict keys: {}".format(output_dict.keys()))
    for key, coef in zip(output_dict.keys(), model.coef_):
        print("{}: {}".format(key, coef))
    # print("The parameters are: {}".format(model.coef_))
    print("The bias is {}".format(model.intercept_))


if __name__ == '__main__':
    master_analyzer()
