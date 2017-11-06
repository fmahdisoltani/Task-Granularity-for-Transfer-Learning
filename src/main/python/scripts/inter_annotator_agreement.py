import os

import numpy as np
import pandas as pd

import torch

from collections import Counter, namedtuple

from scipy.stats import kendalltau, pearsonr, spearmanr, zscore
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable

from ptcap.data.tokenizer import Tokenizer
from ptcap.scores import caption_accuracy, first_token_accuracy, token_accuracy
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.metrics import MultiScorer
from pycocoevalcap.rouge.rouge import Rouge

NUM = 0
LINES = 11
PATH = "/home/waseem/Downloads/Caption_Evaluation_"


def correlation_summary(metrics_dict, final_score, final_actions, final_objects,
                        correlation_metrics):

    corr_dict = get_correlations(metrics_dict, final_score, correlation_metrics)

    print("\nThe correlations with the metrics are:")
    print_dict(corr_dict)

    corr_actions_dict = get_correlations(metrics_dict, final_actions, correlation_metrics)
    print("\nThe correlations with the actions are:")
    print_dict(corr_actions_dict)

    corr_objects_dict = get_correlations(metrics_dict, final_objects, correlation_metrics)
    print("\nThe correlations with the objects are:")
    print_dict(corr_objects_dict)


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
    annotations = {"s1": [], "s2": [], "action": [], "object": []}
    for line in csv_file.itertuples():
        col3, col4 = preprocess(line[3], line[4])
        annotations["s1"].append(line[1])
        annotations["s2"].append(line[2])
        annotations["action"].append(col3)
        annotations["object"].append(col4)
    return annotations


def get_correlations(metrics_dict, final_score, corr_score_dict):
    corr_dict = {}
    for metric in metrics_dict:
        corr_dict[metric] = {}
        for name, corr in corr_score_dict.items():
            if "Kappa" not in name:
                corr_dict[metric][name] = corr(metrics_dict[metric], final_score)[0]
    return corr_dict


def get_scores(metric_scores, key=None):
    scores_dict = {}
    for metric_name in metric_scores:
        metric = metric_scores[metric_name]
        scores = []
        for category in metric:
            if key is None or key in category:
                scores.append(metric[category])
        scores_dict[metric_name] = np.mean(scores)
    return scores_dict


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
    author1_dict = {"action": [], "object": []}
    author2_dict = {"action": [], "object": []}
    sentences = {"s1": [], "s2": []}
    correlation_metrics = {"Kappa": cohen_kappa_score, "Pearson": pearsonr,
                           "Kendall": kendalltau, "Spearman": spearmanr}
    metric_scores = {metric_name: {} for metric_name in correlation_metrics}
    for i in range(1, 11):

        # if i == 9 or i == 3:
        #     continue

        author1 = parse_file("1", i)
        if author1 == -1:
            print("No annotations by author1")
            continue
        else:
            update_dict(global_dict, author1)
            update_dict(author1_dict, author1)
            update_dict(sentences, author1)

        author2 = parse_file("2", i)
        if author2 == -1:
            print("No annotations by author2")
            continue
        else:
            update_dict(global_dict, author2)
            update_dict(author2_dict, author2)

        for category in global_dict.keys():
            print("For " + str(i) + " - " + category)
            for metric_name, metric in correlation_metrics.items():
                score = metric(author1[category], author2[category])
                try:
                    unit = score[0]
                except IndexError:
                    unit = score
                if category + str(i) not in metric_scores[metric_name]:
                    metric_scores[metric_name][category + str(i)] = [unit]
                else:
                    metric_scores[metric_name][category + str(i)].append(unit)
            current_scores = {metric_name: metric_scores[metric_name][category + str(i)][0]
                              for metric_name in metric_scores.keys()}
            print("The scores between author1 and author2 is {}".format(
                  current_scores))

    action_scores = get_scores(metric_scores, "action")
    object_scores = get_scores(metric_scores, "object")
    average_scores = get_scores(metric_scores)

    print("\nAction scores: {}".format(action_scores))
    print("Object scores: {}".format(object_scores))

    print("\nAverage scores: {}".format(average_scores))

    agg_action, agg_objects, agg_all = get_stats(global_dict)

    author1_metric, author2_metric, final_actions, final_objects, final_score = normalize(author1_dict, author2_dict)

    for name, corr in correlation_metrics.items():
        if name is not "Kappa":
            print("\nThe " + name + " score is: {}".format(
                corr(author1_metric, author2_metric)))

    accuracy_dict = try_accuracy(sentences["s1"], sentences["s2"])
    correlation_summary(accuracy_dict, final_score, final_actions, final_objects,
                        correlation_metrics)

    metrics_dict = try_metrics(sentences["s1"], sentences["s2"])
    correlation_summary(metrics_dict, final_score, final_actions, final_objects,
                        correlation_metrics)

    # bubble_chart(agg_all)
    # bar_plot(final_score)
    # bar_plot(agg_objects)

    # matrix = create_2d_array(agg_all)
    # plot_heatmap(matrix)


def normalize(author1_dict, author2_dict):
    action1 = np.array(author1_dict["action"])
    object1 = np.array(author1_dict["object"])

    action2 = np.array(author2_dict["action"])
    object2 = np.array(author2_dict["object"])

    concat_actions = np.concatenate([action1, action2], axis=0)
    concat_objects = np.concatenate([object1, object2], axis=0)

    norm_actions = zscore(concat_actions)
    norm_objects = zscore(concat_objects)

    mean_actions = np.mean([action1, action2], axis=0)
    mean_objects = np.mean([object1, object2], axis=0)

    norm_all_actions = zscore(mean_actions)
    norm_all_objects = zscore(mean_objects)

    metric = norm_all_actions + norm_all_objects

    l = int(len(concat_actions)/2)

    norm_dict1 = {"action": norm_actions[:l], "object": norm_objects[:l]}
    norm_dict2 = {"action": norm_actions[l:], "object": norm_objects[l:]}

    norm_metric1 = norm_dict1["action"] + norm_dict1["object"]
    norm_metric2 = norm_dict2["action"] + norm_dict2["object"]

    return norm_metric1, norm_metric2, norm_all_actions, norm_all_objects, metric


def parse_file(folder_num, index):
    path = os.path.join(PATH + folder_num, str(index) + ".csv")
    if os.path.exists(path):
        csv_file = pd.read_csv(path, header=None)
        annotations = get_annotations(csv_file[11:])
        return annotations
    return -1


def preprocess(str_num1, str_num2):
    num1 = float(str_num1)
    num2 = float(str_num2)
    # if num1 == 0.5 and num2 == 0:
    #     num1 = 0.0
    # if num1 == 0.5 and num2 == 1:
    #     num1 = 1.0
    if num1 == 0.5:
        num1 = 1.0
    if num1 == 1.5:
        num1 = 2.0
    if num2 == 0.5:
        num2 = 0.0
    if num2 == 1.5:
        num2 = 1.0
    if num2 == 3:
        num2 = 2.0
    if num2 == 4:
        num2 = 5.0
    if num2 == 5.0:
        num2 = 3.0
    # if num2 == 3 and num1 <= 1:
    #     num2 = 2.0
    # if num2 == 3 and num1 >= 2:
    #     num2 = 4.0
    # if num2 == 4 and num1 <= 1:
    #     num2 = 4.0
    return num1, num2


def print_dict(some_dict):
    for key in some_dict:
        print(key + ": " + str(some_dict[key]) + "\n")


def try_accuracy(captions, predictions):
    accuracies = {"caption_accuracy": caption_accuracy,
                  "first_accuracy": first_token_accuracy,
                  "accuracy": token_accuracy}
    output_dict = {}
    ScoreAttr = namedtuple("ScoresAttr", "captions predictions")

    tokenizer = Tokenizer()
    tokenizer.load_dictionaries("/home/waseem/Models/")

    for caption, prediction in zip(captions, predictions):
        encoded_caption = Variable(torch.LongTensor([tokenizer.encode_caption(caption)]))
        encoded_prediction = Variable(torch.LongTensor([tokenizer.encode_caption(prediction)]))
        in_tuple = ScoreAttr(encoded_caption, encoded_prediction)
        output_val = {name: [value(in_tuple)[name]] for name, value in accuracies.items()}
        if output_dict == {}:
            output_dict = output_val
        else:
            update_dict(output_dict, output_val)
    return output_dict


def try_metrics(captions, predictions):
    multi_scorer = MultiScorer(BLEU=Bleu(4), ROUGE_L=Rouge(), METEOR=Meteor())

    output_dict = {}

    for caption, prediction in zip(captions, predictions):
        output_val = multi_scorer.score((caption,), [prediction])
        if output_dict == {}:
            output_dict = {key: [value] for key, value in output_val.items()}
        else:
            update_dict(output_dict, output_val)
    return output_dict


def update_dict(global_dict, author):
    for category, scores in author.items():
        if category in global_dict:
            if isinstance(scores, list):
                global_dict[category].extend(scores)
            elif isinstance(scores, float):
                global_dict[category].append(scores)
            else:
                raise NotImplemented


if __name__ == '__main__':
    master_method()
