from constants import *

from sklearn.cluster import KMeans as k
from sklearn import metrics
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import collections

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata


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

def read_sentiment_files(n_segments):
    result = []
    with open('../sentiments/sent_{}.txt'.format(n_segments), 'r', encoding=FILE_ENCODING) as plain, open('../sentiments/sent_{}_savgol.txt'.format(n_segments), 'r', encoding=FILE_ENCODING) as smooth:
        for line in smooth:
            tokens = line.split(SEPARATOR)
            summary = eval(tokens[1])
            result.append(summary[0:n_segments])
    return np.array(result)

def cluster_storylines(clusters, data):
    kmeans = k(n_clusters = clusters)
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

def graph_results(data, assignments, centroids, num_segments):
    centroid_to_data = assignment_dict(assignments, data)
    for centroid, movies in centroid_to_data.items():
        plt.gcf().clear()
        plt.xlabel('Segment of Screenplay')
        plt.ylabel('Average Dialogue Sentiment')
        plt.title('Story Centroid and Assigned Screenplays')
        axes = plt.gca()
        axes.set_ylim([-1.5,1.5])
        for movie in movies[:10]:
            plt.plot(movie, color="orange")
        plt.plot(centroids[centroid], 'k--')
        plt.savefig("../plots/cluster_{}_segments_{}of{}.png".format(num_segments, centroid + 1, len(centroids)))

    plt.gcf().clear()
    for centroid in centroids:
        plt.xlabel('Segment of Screenplay')
        plt.ylabel('Average Dialogue Sentiment')
        plt.title('Story Centroids for 10-Means')
        axes = plt.gca()
        axes.set_ylim([-1.5,1.5])
        plt.plot(centroid)
        plt.savefig("../plots/cluster_{}_segments_centroids.png".format(num_segments))

if __name__ == "__main__":
    max_means   = 12
    segment_nums = [8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400]
    # silhouette_method3(max_means, segment_nums)

    all_scores = []
    for segments in segment_nums:
        data = read_sentiment_files(segments)
        # elbow_method(data, max_means, segments)
        scores = silhouette_method(data, max_means, segments)
        all_scores.append(scores)

        # best = int(input("Best number of means for {} segments? ".format(segments)))
        #
        # assignments, centroids, labels = cluster_storylines(best, data)
        # squared_error = get_squared_error(centroids, assignment_dict(assignments, data))
        #
        # print ("Sum of squared error for {} segments and {} means: {}".format(segments, best, squared_error))
        # graph_results(data, assignments, centroids, segments)
        print ("Done with {}...".format(segments))

    figure = plt.gcf()
    figure.clear()
    figure.set_size_inches(6, 8)
    plt.figure(figsize=(6.5, 15))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    plt.xlabel('K')
    plt.ylabel('Average Silhoette Score')
    plt.title('K vs. Average Silhouette Score\nfor K-Means Clustering')
    plt.ylim(0, 0.4)
    plt.xlim(2, 13)
    count = 0
    prev_y = None
    for scores, segments in zip(all_scores, segment_nums):
        plt.plot(range(2,max_means+1), scores, c=tableau20[count])
        # plt.scatter(range(2,max_means), scores, c=tableau20[count])
        y_pos = scores[-1]
        if prev_y:
            if abs(prev_y - y_pos) < 0.005:
                print ('changing for {}'.format(segments))
                y_pos = y_pos - 0.005 * (-1 if prev_y - y_pos < 0 else 1)
        plt.text(12.2, y_pos, "{} Segments".format(segments), color=tableau20[count])
        count += 1
        prev_y = y_pos
    plt.savefig('../plots/km_silhouette_summary.png')   
