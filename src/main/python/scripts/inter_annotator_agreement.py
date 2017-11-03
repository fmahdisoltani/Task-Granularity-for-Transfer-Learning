import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
import sklearn.metrics as metrics

from collections import Counter

LINES = 11
PATH = "/home/waseem/Downloads/Caption_Evaluation_"


def visualize(annotations_dict):
    action = annotations_dict["action"]
    object = annotations_dict["object"]

    agg_action = Counter(action)
    agg_object = Counter(object)

    ax = sns.barplot(x=[k for k in agg_action.keys()],
                     y=[v for v in agg_action.values()])
    plt.show(ax)


def get_annotations(csv_file):
    annotations = {"action": [], "object": []}
    for line in csv_file.itertuples():
        annotations["action"].append(line[3])
        annotations["object"].append(line[4])
    return annotations


def parse_file(folder_num, index):
    path = os.path.join(PATH + folder_num, str(index) + ".csv")
    if os.path.exists(path):
        csv_file = pd.read_csv(path, header=None)
        annotations = get_annotations(csv_file[11:])
        return annotations
    return -1

if __name__ == '__main__':
    global_dict = {"action": [], "object": []}
    kappa_score = {}
    correlation = {}
    for i in range(10):
        author1 = parse_file("1", i + 1)
        if author1 == -1:
            print("No annotations by author1")
            continue
        else:
            for category, score in author1.items():
                global_dict[category].extend(score)
            author2 = parse_file("2", i + 1)
        if author2 == -1:
            print("No annotations by author2")
            continue
        else:
            for category, score in author2.items():
                global_dict[category].extend(score)

        for category in author1.keys():
            kappa_score_ = metrics.cohen_kappa_score(author1[category],
                                                    author2[category])
            correlation_ = scipy.stats.spearmanr(author1[category],
                                               author2[category])
            kappa_score[category + str(i)] = kappa_score_
            correlation[category + str(i)] = correlation_

            print("For " + str(i + 1) + " - " + category)
            print("The kappa score between author1 and author2 is {}".format(kappa_score_))
            print("The spearman correlation between author1 and author2 is {}".format(
                correlation_))

    visualize(global_dict)