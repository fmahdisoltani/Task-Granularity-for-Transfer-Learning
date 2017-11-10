import pickle
import json
import os
import time

import numpy as np
import pandas as pd

import torch

from collections import Counter, namedtuple

from gensim.models import KeyedVectors
from scipy.stats import kendalltau, pearsonr, spearmanr, zscore
from sklearn.metrics import cohen_kappa_score
from torch.autograd import Variable

from ptcap.data.tokenizer import Tokenizer
from ptcap.scores import (LCS, caption_accuracy, first_token_accuracy, fscore,
                          gmeasure, token_accuracy)
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.metrics import MultiScorer
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice

NUM = 0
LINES = 11
PATH = "/home/waseem/Downloads/Caption_Evaluation_"


def correlation_summary(metrics_dict, final_score, final_actions, final_objects,
                        correlation_metrics):

    corr_dict = get_correlations(metrics_dict, final_score, correlation_metrics)

    metrics_df = pd.DataFrame(data=corr_dict)
    metrics_df.to_csv("/home/waseem/Metrics Analysis/metrics_metric.csv")

    print("\nThe correlations with the metrics are:")
    print_dict(corr_dict)

    corr_actions_dict = get_correlations(metrics_dict, final_actions, correlation_metrics)

    actions_df = pd.DataFrame(data=corr_actions_dict)
    actions_df.to_csv("/home/waseem/Metrics Analysis/metrics_actions.csv")

    print("\nThe correlations with the actions are:")
    print_dict(corr_actions_dict)

    corr_objects_dict = get_correlations(metrics_dict, final_objects, correlation_metrics)

    objects_df = pd.DataFrame(data=corr_objects_dict)
    objects_df.to_csv("/home/waseem/Metrics Analysis/metrics_objects.csv")

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
    if isinstance(metrics_dict, dict):
        for metric in metrics_dict:
            corr_dict[metric] = {}
            for name, corr in corr_score_dict.items():
                if "Kappa" not in name:
                    corr_dict[metric][name] = corr(metrics_dict[metric], final_score)[0]
    elif isinstance(metrics_dict, list):
        for name, corr in corr_score_dict.items():
            if "Kappa" not in name:
                corr_dict[name] = corr(metrics_dict, final_score)[0]
    return corr_dict


def get_encodings(accuracies, caption, prediction, tokenizer, score_attr):
    encoded_caption = Variable(
        torch.LongTensor([tokenizer.encode_caption(caption)]))
    encoded_prediction = Variable(
        torch.LongTensor([tokenizer.encode_caption(prediction)]))
    in_tuple = score_attr([caption], encoded_caption, encoded_prediction)
    lcs_output = accuracies["LCS"](in_tuple)

    accuracy_dict = {name: [value(in_tuple)[name]] for name, value in
                  accuracies.items()
                  if name is not "LCS"}
    accuracy_dict.update({name: [value] for name, value in lcs_output.items()})
    return accuracy_dict


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


def get_wmd(model, caption, prediction):
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    cap = [w for w in caption.split() if (w in model and w not in stop_words)]
    pred = [w for w in prediction.split() if (w in model and w not in stop_words)]
    wmd = model.wmdistance(cap, pred)
    return {"wmd": [-1*wmd]}


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

    (author1_metric, author2_metric, final_actions, final_objects,
     final_score) = normalize(author1_dict, author2_dict)

    bins1 = split_into_bins(author2_metric, 1)
    bins2 = split_into_bins(author2_metric, 2)

    for bin_name1, bin1 in bins1.items():
        for bin_name2, bin2 in bin2.items():
            corr = get_correlations(scores, gt, correlation_metrics)


    corr_bin_dict = {}

    for bin_name, bin in bins.items():
        gt = [author1_metric[i] for i in bin]
        scores = [author2_metric[i] for i in bin]
        corr = get_correlations(scores, gt, correlation_metrics)
        # scatter_plot(gt, scores)
        print("Correlations are {}".format(corr))
        if bin_name not in corr_bin_dict:
            corr_bin_dict[bin_name] = corr
        else:
            corr_bin_dict[bin_name] = corr
    print(corr_bin_dict)
    # corr = get_correlations(output_dict["METEOR"], final_score, correlation_metrics)
    # print("Correlations are {}".format(corr))

    # with open("/home/waseem/Metrics Analysis/author_based", "wb") as f:
    #     pickle.dump((author1_metric, author2_metric, final_actions, final_objects,
    #                  final_score), f)

    # for name, corr in correlation_metrics.items():
    #     if name is not "Kappa":
    #         print("\nThe " + name + " score is: {}".format(
    #             corr(author1_metric, author2_metric)))

    # metrics_dict = try_metrics(sentences["s1"], sentences["s2"])
    # correlation_summary(metrics_dict, final_score, final_actions, final_objects,
    #                     correlation_metrics)

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


def try_metrics(captions, predictions):

    print("Loading Word Vectors...")
    a = time.time()
    model = KeyedVectors.load_word2vec_format(
        '/home/waseem/Models/GoogleNews-vectors-negative300.bin', binary=True)
    b = time.time()
    print("Word Vectors loaded in {}".format(b - a))

    tokenizer = Tokenizer()
    tokenizer.load_dictionaries("/home/waseem/Models/")

    accuracies = {"caption_accuracy": caption_accuracy,
                  "first_accuracy": first_token_accuracy,
                  "accuracy": token_accuracy,
                  "LCS": LCS([fscore, gmeasure], tokenizer)}

    ScoreAttr = namedtuple("ScoresAttr", "string_captions captions predictions")
    multi_scorer = MultiScorer(BLEU=Bleu(4), ROUGE_L=Rouge(), METEOR=Meteor(),
                               SPICE=Spice())

    output_dict = {}

    a = time.time()

    for caption, prediction in zip(captions, predictions):

        encodings = get_encodings(accuracies, caption, prediction, tokenizer,
                                  ScoreAttr)
        output_val = encodings

        multi_scores = multi_scorer.score((caption,), [prediction])
        output_val.update({key: [value] for key, value in multi_scores.items()})

        wmd = get_wmd(model, caption, prediction)
        output_val.update(wmd)

        if output_dict == {}:
            output_dict = output_val
        else:
            update_dict(output_dict, output_val)
            # with open("/home/waseem/Metrics Analysis/metric_values", "w") as f:
            #     json.dump(output_dict, f)

    b = time.time()

    print("try_metrics took {}".format(b - a))

    # with open("/home/waseem/Metrics Analysis/metric_values", "w") as f:
    #     json.dump(output_dict, f)

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
