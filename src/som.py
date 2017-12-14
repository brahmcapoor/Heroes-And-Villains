from constants import *

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from minisom import MiniSom

import argparse
import collections

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def load_sentiment_data(filename):
	sentiment_data = []
	with open(filename, 'r') as f:
		for line in f:
			_, sentiments = line.split(SEPARATOR)
			sentiment_data.append(eval(sentiments))
	return np.array(sentiment_data)			

def train_som(k, sentiment_data):
	som = MiniSom(2*k, 2*k, sentiment_data.shape[1])
	print("Training SOM...", end="\r")
	som.train_random(sentiment_data, 100)
	print("Training complete!")
	return som

def map_sentiment_to_assign(sentiments_file, assignments_file):
	res = {}

	with open(sentiments_file, 'r') as sentiments, open(assignments_file, 'r') as assignments:
		for s, a in zip(sentiments, assignments):
			sentiment = tuple(eval(s.split(SEPARATOR)[1]))
			assign = int(a.split(SEPARATOR)[1])
			res[sentiment] = assign

	return res


def plot_som(k, som, sentiment_to_assign):
	plt.figure(figsize=(2*k, 2*k))

	cm = plt.get_cmap('nipy_spectral')
	colors = [ cm( float(i) / k) for i in range(k) ]

	num_assigned = collections.defaultdict(lambda : [0, [0, 0, 0], [0] * k])

	for sent, assign in sentiment_to_assign.items():
		w = som.winner(sent)
		num_assigned[w][0] += 1
		color = tableau20[assign]
		for i in range(3):
			num_assigned[w][1][i] += color[i]
		num_assigned[w][2][assign - 1] += 1


	for w in num_assigned:
		n = num_assigned[w][0] * 1.0
		num_assigned[w][1] = tuple([c / n for c in num_assigned[w][1]])
		num_assigned[w][2] = np.argmax(num_assigned[w][2])

	for w in num_assigned:
		n, color, assign = num_assigned[w]
		plt.scatter(w[0] + 0.5, w[1] + 0.5, c=color, s=n*500, marker='o', alpha=0.7)
		plt.text(w[0] + 0.45, w[1] + 0.45, s=str(assign))


	plt.title('Self Organizing Map Representation of {}-Means Cluster Assignments'.format(k), fontsize=20)
	plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]]) 
	plt.xticks([], [])
	plt.yticks([], [])
	plt.savefig("SOM.png")


def make_som(k, filter, n_segments):
	sentiments_file = SENTIMENT_OUTPUT_PATH.format(n_segments, filter)
	sentiment_data = load_sentiment_data(sentiments_file)
	som = train_som(k, sentiment_data)

	assignments_file = CLUSTER_OUTPUT_ASSIGNMENT_PATH.format(n_segments, k, filter)
	sentiment_to_assign = map_sentiment_to_assign(sentiments_file, assignments_file)
	plot_som(k, som, sentiment_to_assign)

if __name__ == "__main__":
	p = argparse.ArgumentParser(description='Generate SOM visualizations')

	p.add_argument("-k", "--k_means", type=int, required=True)
	p.add_argument("-f", "--filter", type=str, 
				   choices=['plain', 'savgol', 'slide'], 
				   required=True)
	p.add_argument("-s", "--n_segments", 
				   type=int, 
				   choices=SEGMENT_SIZES,
				   required=True)

	args = p.parse_args()

	make_som(k = args.k_means,
			 filter = args.filter,
			 n_segments = args.n_segments)