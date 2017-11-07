import json
import pickle

import matplotlib.pyplot as plt
import seaborn as sns


def scatter_plot(x, y):
    plt.scatter(x, y)
    plt.show()


def master_analyzer():

    with open("/home/waseem/Metrics Analysis/author_based", "rb") as f:
        (author1_metric, author2_metric, final_actions, final_objects,
         final_score) = pickle.load(f)

    with open("/home/waseem/Metrics Analysis/metric_values", "w") as f:
        output_dict = json.load(f)

    scatter_plot(final_score, output_dict["METEOR"])


if __name__ == '__main__':
    master_analyzer()
