import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, zscore
import seaborn as sns
import sklearn.metrics as metrics

from collections import Counter

NUM = 0
LINES = 11
PATH = "/home/waseem/Downloads/Caption_Evaluation_"


def bar_plot(aggregate):
    try:
        ax = sns.barplot(x=[k for k in aggregate.keys()],
                         y=[v for v in aggregate.values()])
    except:
        c = Counter(aggregate)
        ax = sns.barplot(x=list(c.keys()), y=list(c.values()))
    plt.show(ax)


def bubble_chart(aggregate):
    x, y, z = [], [], []
    for pair, count in aggregate.items():
        x.append(pair[0])
        y.append(pair[1])
        z.append(count*4)
    plt.scatter(x,y, s=z, alpha=0.5)
    plt.xlabel("Action Scores")
    plt.ylabel("Object Scores")
    plt.show()


def create_2d_array(counter):
    x = []
    y = []

    for pair in counter.keys():
        if pair[0] not in x:
            x.append(pair[0])
        if pair[1] not in y:
            y.append(pair[1])

    x = np.sort(x)
    y = np.sort(y)

    matrix = np.zeros([len(x), len(y)])

    for i in range(len(x)):
        for j in range(len(y)):
            matrix[i][j] = np.log(counter[(i, j)])

    return matrix


def get_annotations(csv_file):
    annotations = {"action": [], "object": []}
    for line in csv_file.itertuples():
        v1, v2 = preprocess(line[3], line[4])
        annotations["action"].append(v1)
        annotations["object"].append(v2)
    return annotations


def get_stats(annotations_dict):
    action = annotations_dict["action"]
    object = annotations_dict["object"]

    agg_action = Counter(action)
    agg_object = Counter(object)

    action_percentages = {}
    object_percentages = {}
    for action_, count in agg_action.items():
        action_percentages[action_] = count/np.sum(list(agg_action.values()))

    for object_, count in agg_object.items():
        object_percentages[object_] = count/np.sum(list(agg_object.values()))

    agg_all = Counter(zip(action, object))

    return agg_action, agg_object, agg_all


def master_method():
    global_dict = {"action": [], "object": []}
    kappa_score = {}
    correlation = {}
    author1_dict = {"action": [], "object": []}
    author2_dict = {"action": [], "object": []}
    for i in range(1, 11):

        author1 = parse_file("1", i)
        if author1 == -1:
            print("No annotations by author1")
            continue
        else:
            update_dict(global_dict, author1)
            update_dict(author1_dict, author1)

        author2 = parse_file("2", i)
        if author2 == -1:
            print("No annotations by author2")
            continue
        else:
            update_dict(global_dict, author2)
            update_dict(author2_dict, author2)

        for category in author1.keys():
            # kappa_score_ = metrics.cohen_kappa_score(author1[category],
            #                                          author2[category])
            correlation_ = spearmanr(author1[category],
                                                 author2[category])
            # kappa_score[category + str(i)] = kappa_score_
            correlation[category + str(i)] = correlation_.correlation

            print("For " + str(i) + " - " + category)
            # print(
            #     "The kappa coefficient between author1 and author2 is {}".format(
            #         kappa_score_))
            print(
                "The spearman correlation between author1 and author2 is {}".format(
                    correlation_))

    action_kappa = [value for key, value in kappa_score.items() if "action" in key]
    object_kappa = [value for key, value in kappa_score.items() if "object" in key]

    action_spearman = [value for key, value in correlation.items() if "action" in key]
    object_spearman = [value for key, value in correlation.items() if "object" in key]

    print("\nAction kappa coefficient: {}".format(np.mean(action_kappa)))
    print("Object kappa coefficient: {}".format(np.mean(object_kappa)))

    print("\nAction correlations: {}".format(np.mean(action_spearman)))
    print("Object correlations: {}".format(np.mean(object_spearman)))

    print("\nAverage kappa coefficient: {}".format(np.mean(list(kappa_score.values()))))
    print(
        "Average correlations: {}".format(np.mean(list(correlation.values()))))
    agg_action, agg_objects, agg_all = get_stats(global_dict)

    # author1_metric, author2_metric = regress(author1_dict, author2_dict, 0)
    author1_metric, author2_metric = normalize(author1_dict, author2_dict)

    print("The metric correlation is: {}".format(spearmanr(author1_metric, author2_metric)))
    # bubble_chart(agg_all)
    # bar_plot(agg_action)
    # bar_plot(agg_objects)

    # matrix = create_2d_array(agg_all)
    # plot_heatmap(matrix)


def normalize(author1_dict, author2_dict):
    action1 = np.array(author1_dict["action"])
    object1 = np.array(author1_dict["object"])

    action2 = np.array(author2_dict["action"])
    object2 = np.array(author2_dict["object"])

    actions = np.concatenate([action1, action2], axis=0)
    objects = np.concatenate([object1, object2], axis=0)

    norm_actions = zscore(actions)
    norm_objects = zscore(objects)

    l = int(len(actions)/2)

    norm_dict1 = {"action": norm_actions[:l], "object": norm_objects[:l]}
    norm_dict2 = {"action": norm_actions[l:], "object": norm_objects[l:]}

    norm_metric1 = norm_dict1["action"] + norm_dict1["object"]
    norm_metric2 = norm_dict2["action"] + norm_dict2["object"]

    return norm_metric1, norm_metric2


def parse_file(folder_num, index):
    path = os.path.join(PATH + folder_num, str(index) + ".csv")
    if os.path.exists(path):
        csv_file = pd.read_csv(path, header=None)
        annotations = get_annotations(csv_file[11:])
        return annotations
    return -1


def plot_heatmap(matrix):
    sns.heatmap(matrix, annot=True)
    plt.show()


def preprocess(str_num1, str_num2):
    num1 = float(str_num1)
    num2 = float(str_num2)
    # if num1 == 0.5 and num2 == 0:
    #     num1 = 0.0
    # if num1 == 0.5 and num2 == 1:
    #     num1 = 1.0
    # if num1 == 1.5:
    #     num1 = 2.0
    # if num2 == 0.5:
    #     num2 = 0.0
    # if num2 == 1.5:
    #     num2 = 2.0
    # if num2 == 3 and num1 <= 1:
    #     num2 = 2.0
    # if num2 == 3 and num1 >= 2:
    #     num2 = 4.0
    # if num2 == 4 and num1 <= 1:
    #     num2 = 4.0
    return num1, num2


def regress(author1_dict, author2_dict, w):
    # action1 = np.array(author1_dict["action"])
    # object1 = np.array(author1_dict["object"])
    #
    # action2 = np.array(author2_dict["action"])
    # object2 = np.array(author2_dict["object"])
    #
    # norm_action1 = (action1 - np.mean(action1))/np.std(action1)
    # norm_object1 = (object1 - np.mean(object1))/np.std(object1)

    # bar_plot(norm_object1)
    #
    # norm_action2 = (action2 - np.mean(action2))/np.std(action2)
    # norm_object2 = (object2 - np.mean(object2))/np.std(object2)
    #
    # num = np.sum((norm_action1 - norm_action2) * (norm_object2 - norm_object1))
    # den = np.sum(np.square(norm_object1 - norm_object2))
    #
    # output = num/den
    # print("w*: {}".format(output))

    # w = output

    # author1_score = action1 + w * object1
    # author2_score = action2 + w * object2
    #
    # a1_s = norm_action1 + w * norm_object1
    # a2_s = norm_action2 + w * norm_object2

    # diff = np.square(author1_score - author2_score)
    # y = np.sum(diff)

    # author1_score = np.round(author1_score)
    # author2_score = np.round(author2_score)
    #
    # y1 = metrics.cohen_kappa_score(author1_score, author2_score)

    # y1 = scipy.stats.spearmanr(a1_s, a2_s)

    # return norm_object1 + norm_action1, norm_object2 + norm_action2
    return 0


def scatter_plot(x, y):
    plt.scatter(x, y)
    plt.show()


def update_dict(global_dict, author):
    for category, scores in author.items():
        global_dict[category].extend(scores)

if __name__ == '__main__':
    master_method()
