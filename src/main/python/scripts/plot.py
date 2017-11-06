import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


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


def plot_heatmap(matrix):
    sns.heatmap(matrix, annot=True)
    plt.show()


def scatter_plot(x, y):
    plt.scatter(x, y)
    plt.show()


#######################################################
#######################################################


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
