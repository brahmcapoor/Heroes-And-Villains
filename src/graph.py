import dill
from sklearn.cluster import KMeans
import numpy as np
from constants import *
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import collections
import argparse

#Loads dicts and runs k_means on either individual/all movies
def sentiment_clustering(k_clusters, plot_type):
	avg_sentiments = load_from_pkl('avg_sentiment.pkl')
	protag_sentiments = load_from_pkl('protagonist_sentiment.pkl')
	movieids_to_names =  load_from_pkl('movieids_to_names.pkl')
	charids_to_names = load_from_pkl('charids_to_names.pkl')

	n_clusters_to_sil_score = collections.defaultdict(float)
	sentiments_to_names = {}
	numMovies = 0
	all_char_sentiments = []

	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		for movie in dataset:
			char_sentiments = []
			movie = movie.strip().split(SEPARATOR)
			m_id, protag_id = movie[0], movie[1]
			cur_avg_sentiments = avg_sentiments[m_id]
			cur_protag_sentiments = protag_sentiments[m_id]
			if len(cur_protag_sentiments) < 5:
				continue
			numMovies += 1
			for k,v in cur_avg_sentiments.items():
				if k != protag_id:
					cluster_val = [v[0],cur_protag_sentiments[k][0]]
					char_sentiments.append(cluster_val)
					sentiments_to_names[tuple(cluster_val)] = charids_to_names[k]

			all_char_sentiments.extend(char_sentiments)
			X = np.array(char_sentiments)
			scores = cluster_silhouette_scores(X, movieids_to_names[m_id])

			if plot_type == 'ind':
				run_kMeans_single_movie(k_clusters, X, sentiments_to_names, movieids_to_names[m_id])
				plot_individual_silhouette_scores(scores, movieids_to_names[m_id])

			else:
				for i, score in enumerate(scores):
					n_clusters_to_sil_score[i] += score

	if plot_type == 'all':
		avg_scores = []
		for k,v in n_clusters_to_sil_score.items():
			n_clusters_to_sil_score[k] = v/numMovies #divide score by number of movies
			avg_scores.append(n_clusters_to_sil_score[k])

		X_all = np.array(all_char_sentiments)

		run_kMeans_all_movies(k_clusters, X_all, sentiments_to_names)
		plot_overall_silhouette_scores(avg_scores)

def load_from_pkl(filename):
	pkld_obj = None
	with open(filename, 'rb') as f:
		pkld_obj = dill.load(f)
		f.close()
	return pkld_obj

def run_kMeans_single_movie(k, X, sentiments_to_names, movie_name):
	kmeans = KMeans(n_clusters=k).fit(X)
	print(kmeans.cluster_centers_)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
	print(X)

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
						 extent=(xx.min(), xx.max(), yy.min(), yy.max()),
						 cmap=plt.cm.Paired,
						 aspect='auto', origin='lower')
	plt.xlabel('Average Sentiment')
	plt.ylabel('Protagonist Sentiment Towards')

	colors = ['k', 'g', 'r', 'c', 'm', 'y', 'b',]
	styles = ['D', 'o', 'v', '*', '+', '3', '4']

	for i in range(len(X)):
		cur_label = sentiments_to_names[(X[i,0], X[i,1])]
		plt.plot(X[i,0], X[i,1], styles[i//7]+colors[i%7], markersize=5, label=cur_label)

	plt.legend(bbox_to_anchor=(.96,1))
	centroids = kmeans.cluster_centers_
	plt.title('{}-Means Clustering on {}'.format(k, movie_name.title()))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()

def run_kMeans_all_movies(k, X, sentiments_to_names):
	kmeans = KMeans(n_clusters=k).fit(X)
	print(kmeans.cluster_centers_)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
						 extent=(xx.min(), xx.max(), yy.min(), yy.max()),
						 cmap=plt.cm.Paired,
						 aspect='auto', origin='lower')
	plt.xlabel('Average Sentiment')
	plt.ylabel('Protagonist Sentiment Towards')

	for i in range(len(X)):
		cur_label = sentiments_to_names[(X[i,0], X[i,1])]
		if kmeans.labels_[i] == 0:
			plt.plot(X[i,0], X[i,1], 'ko', markersize=2, label=cur_label)
		elif kmeans.labels_[i] == 1:
			plt.plot(X[i,0], X[i,1], 'bo', markersize=2, label=cur_label)
		elif kmeans.labels_[i] == 2:
			plt.plot(X[i,0], X[i,1], 'co', markersize=2, label=cur_label)
		elif kmeans.labels_[i] == 3:
			plt.plot(X[i,0], X[i,1], 'yo', markersize=2, label=cur_label)
		else:
			plt.plot(X[i,0], X[i,1], 'go', markersize=2, label=cur_label)

	# Plot the centroids as a white X
	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
							marker='x', s=169, linewidths=3,
							color='w', zorder=10)
	plt.title('{}-Means Clustering on All Characters'.format(k))
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	plt.show()


def cluster_silhouette_scores(X, movie_name):
	range_n_clusters = [2, 3, 4, 5]
	scores = []
	for n_clusters in range_n_clusters:
		clusterer = KMeans(n_clusters=n_clusters, random_state=10)
		cluster_labels = clusterer.fit_predict(X)

		# The silhouette_score gives the average value for all the samples.
		# This gives a perspective into the density and separation of the formed
		# clusters
		silhouette_avg = silhouette_score(X, cluster_labels)
		scores.append(silhouette_avg)

	return scores

def plot_overall_silhouette_scores(silhoeutte_scores):
	ind = np.arange(4)
	fig, ax = plt.subplots()

	width = .5

	rects = plt.bar(ind, silhoeutte_scores)

	ax.set_xlabel('Number of Clusters (k)')
	ax.set_ylabel('Overall Average Silhouette Scores')
	ax.set_title('Average Silhouette Scores Across All Movies')
	ax.set_xticks(ind)
	ax.set_xticklabels(('2', '3', '4', '5'))

	plt.show()

def plot_individual_silhouette_scores(silhouette_scores, movie_name):
	ind = np.arange(4)
	fig, ax = plt.subplots()

	width = .5

	rects = plt.bar(ind, silhouette_scores, color='r')

	ax.set_xlabel('Number of Clusters (k)')
	ax.set_ylabel('Average Silhouette Scores')
	ax.set_title('Average Silhouette Scores for {}'.format(movie_name.title()))
	ax.set_xticks(ind)
	ax.set_xticklabels(('2', '3', '4', '5'))

	plt.show()
			
	
if __name__ == "__main__":
	p = argparse.ArgumentParser(description='Run K-Means on individual movies or all movies')
	p.add_argument("-k", "--k_means", type=int, choices=[2, 3, 4, 5], required=True)
	p.add_argument("-t", "--type", type=str, choices=['ind', 'all'], required=True)

	args = p.parse_args()
	sentiment_clustering(args.k_means, args.type)
