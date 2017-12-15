from constants import *

from sklearn.cluster import KMeans
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import collections
import itertools
import argparse
import os

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

ax = plt.subplot(111)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)

def read_sentiment_files(n, filter_type):
    result = []
    m_ids = []
    with open(SENTIMENT_OUTPUT_PATH.format(n, filter_type), 'r', encoding=FILE_ENCODING) as f:
        for line in f:
            tokens = line.split(SEPARATOR)
            sentiments = eval(tokens[1])
            result.append(sentiments[0:n])
            m_ids.append(tokens[0])

    return np.array(result), m_ids

def cluster_storylines(k, data):
    kmeans = KMeans(n_clusters = k)
    assignments = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_

    return assignments, centroids, kmeans.labels_

def assignment_dict(assignments, data):
    results = collections.defaultdict(list)
    for assignment, points in zip(assignments, data):
        results[assignment].append(points)
    return results

def get_squared_error(centroids, centroid_to_data):
    def distance(first, second):
        return sum((a - b) ** 2 for a, b in zip(first, second))

    result = 0
    for centroid_index, centroid in enumerate(centroids):
        for point in centroid_to_data[centroid_index]:
            result += distance(point, centroid)
    return result

def elbow_method(data, max_means, segments):
    sum_squared_errors = []

    for k in range(2,max_means):
        assignments, centroids, labels = cluster_storylines(k, data)
        centroid_to_data = assignment_dict(assignments, data)
        squared_error = get_squared_error(centroids, centroid_to_data)
        sum_squared_errors.append(squared_error)

    plt.gcf().clear()
    plt.xlabel('K')
    plt.ylabel('Sum of Squared Errors')
    plt.title('K vs. Sum of Squared Errors for K-Means Clustering')
    plt.plot(range(2,max_means), sum_squared_errors, 'k--')
    plt.scatter(range(2,max_means), sum_squared_errors, c='k')
    plt.savefig('../plots/km_elbow_{}.png'.format(segments))

def silhouette_method(data, max_means, segments):
    silhouette_scores = []

    for k in range(2,max_means+1):
        assignments, centroids, labels = cluster_storylines(k, data)
        centroid_to_data = assignment_dict(assignments, data)
        score = metrics.silhouette_score(data, labels, sample_size=None)
        silhouette_scores.append(score)
    return silhouette_scores

    # plt.gcf().clear()
    # plt.xlabel('K')
    # plt.ylabel('Average Silhoette Score')
    # plt.title('K vs. Average Silhoette Score for K-Means Clustering')
    # plt.plot(range(2,max_means), silhouette_scores, 'k--')
    # plt.scatter(range(2,max_means), silhouette_scores, c='k')
    # plt.savefig('../plots/km_silhouette_{}.png'.format(segments))

def graph_results(data, assignments, centroids, n, filter_type):
    path = CLUSTER_DIR_PATH.format(n, len(centroids), filter_type)
    if not os.path.isdir(path):
        os.makedirs(path)

    centroid_to_data = assignment_dict(assignments, data)
    for centroid, movies in centroid_to_data.items():
        plt.gcf().clear()
        plt.xlabel('Segment of Screenplay')
        plt.ylabel('Average Dialogue Sentiment')
        plt.title('Story Centroid and Assigned Screenplays')
        axes = plt.gca()
        axes.set_ylim([-1.5,1.5])
        for movie in movies[:min(16, len(movies))]:
            plt.plot(movie, color="orange")
        plt.plot(centroids[centroid], 'k--')
        plt.savefig(path + "clusters_{}.png".format(centroid + 1))

    plt.gcf().clear()
    for centroid in centroids:
        plt.xlabel('Segment of Screenplay')
        plt.ylabel('Average Dialogue Sentiment')
        plt.title('Story Centroids for {}-Means'.format(len(centroids)))
        axes = plt.gca()
        axes.set_ylim([-1.5,1.5])
        plt.plot(centroid)
        plt.savefig(path + "centroids.png")

def save_results(m_ids, assignments, centroids):
    path = CLUSTER_DIR_PATH.format(n, len(centroids), filter_type)
    if not os.path.isdir(path):
        os.makedirs(path)

    with open(CLUSTER_OUTPUT_CENTROID_PATH.format(n, len(centroids), filter_type), 'w') as c:
        for centroid in centroids:
            c.write('{}\n'.format(list(centroid)))
        c.flush()

    with open(CLUSTER_OUTPUT_ASSIGNMENT_PATH.format(n, len(centroids), filter_type), 'w') as a:
        for m_id, assignment in zip(m_ids, assignments):
            a.write('{} {} {}\n'.format(m_id, SEPARATOR, assignment))
        a.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-m', '--movie_id', type=str, default='all')
    parser.add_argument('-k', '--k_means', type=int, nargs='+', default=[6])
    parser.add_argument('-s', '--segment_nums', type=int, nargs='+', default=SEGMENT_SIZES)
    parser.add_argument('-f', '--filters', type=str, nargs='+', default=['plain', 'savgol', 'slide'])

    args = parser.parse_args()
    for filter_type in args.filters:
        if filter_type not in ['plain', 'savgol', 'slide']:
            print ("ERROR: invalid filter type \'{}\'".format(args.filter))
            sys.exit(1)

    print ("Clustering on products of {}, {}, {}".format(args.k_means, args.segment_nums, args.filters))
    for k, n, filter_type in tqdm(list(itertools.product(args.k_means, args.segment_nums, args.filters))):
        data, m_ids = read_sentiment_files(n, filter_type)
        assignments, centroids, labels = cluster_storylines(k, data)
        graph_results(data, assignments, centroids, n, filter_type)
        save_results(m_ids, assignments, centroids)


    # max_means = 12
    #
    # all_scores = []
    # for segments in args.segment_nums:
    #     data = read_sentiment_files(segments)
    #     # elbow_method(data, max_means, segments)
    #     scores = silhouette_method(data, max_means, segments)
    #     all_scores.append(scores)
    #
    #     # best = int(input("Best number of means for {} segments? ".format(segments)))
    #     #
    #     # assignments, centroids, labels = cluster_storylines(best, data)
    #     # squared_error = get_squared_error(centroids, assignment_dict(assignments, data))
    #     #
    #     # print ("Sum of squared error for {} segments and {} means: {}".format(segments, best, squared_error))
    #     # graph_results(data, assignments, centroids, segments)
    #     print ("Done with {}...".format(segments))
    #
    # figure = plt.gcf()
    # figure.clear()
    # figure.set_size_inches(6, 8)
    # plt.figure(figsize=(6.5, 15))
    # ax = plt.subplot(111)
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(True)
    # ax.spines["right"].set_visible(False)
    # ax.spines["left"].set_visible(True)
    # plt.xlabel('K')
    # plt.ylabel('Average Silhoette Score')
    # plt.title('K vs. Average Silhouette Score\nfor K-Means Clustering')
    # plt.ylim(0, 0.4)
    # plt.xlim(2, 13)
    # count = 0
    # prev_y = None
    # for scores, segments in zip(all_scores, segment_nums):
    #     plt.plot(range(2,max_means+1), scores, c=tableau20[count])
    #     # plt.scatter(range(2,max_means), scores, c=tableau20[count])
    #     y_pos = scores[-1]
    #     if prev_y:
    #         if abs(prev_y - y_pos) < 0.005:
    #             print ('changing for {}'.format(segments))
    #             y_pos = y_pos - 0.005 * (-1 if prev_y - y_pos < 0 else 1)
    #     plt.text(12.2, y_pos, "{} Segments".format(segments), color=tableau20[count])
    #     count += 1
    #     prev_y = y_pos
    # plt.savefig('../plots/km_silhouette_summary.png')
