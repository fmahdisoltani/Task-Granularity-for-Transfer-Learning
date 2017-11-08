import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import Counter


def get_correlations(metrics_dict, final_score, corr_score_dict):
    corr_dict = {}
    for metric in metrics_dict:
        corr_dict[metric] = {}
        for name, corr in corr_score_dict.items():
            if "Kappa" not in name:
                corr_dict[metric][name] = corr(metrics_dict[metric], final_score)[0]
    return corr_dict


def master_analyzer():

    with open("/home/waseem/Metrics Analysis/author_based", "rb") as f:
        (author1_metric, author2_metric, final_actions, final_objects,
         final_score) = pickle.load(f)

    # with open("/home/waseem/Metrics Analysis/metric_values", "w") as f:
    #     output_dict = json.load(f)

    # scatter_plot(final_score, output_dict["METEOR"])

    scatter_plot(author1_metric, author2_metric, "Author1", "Author2")

    correlation_metrics = {"Kappa": cohen_kappa_score, "Pearson": pearsonr,
                           "Kendall": kendalltau, "Spearman": spearmanr}

    corr_dict = get_correlations(author1_metric, author2_metric, corr_score_dict)

    print("The correlation is: {}".format(corr_dict))

    # scatter_plot(final_objects, final_score)


def scatter_plot(x, y, xlabel, ylabel):
    agg_all = Counter(zip(x, y))
    z = []
    for x_, y_ in zip(x, y):
        z.append(agg_all[x_, y_])

    plt.scatter(x, y, s=z, alpha=0.5)
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


if __name__ == '__main__':
    master_analyzer()
