from constants import *

import numpy as np
from minisom import MiniSom

import argparse

# data = [[ 0.80,  0.55,  0.22,  0.03],
#         [ 0.82,  0.50,  0.23,  0.03],
#         [ 0.80,  0.54,  0.22,  0.03],
#         [ 0.80,  0.53,  0.26,  0.03],
#         [ 0.79,  0.56,  0.22,  0.03],
#         [ 0.75,  0.60,  0.25,  0.03],
#         [ 0.77,  0.59,  0.22,  0.03]]    

# som = MiniSom(6, 6, 4, sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
# print("Training...")
# som.train_batch(data, 100) # trains the SOM with 100 iterations
# print("...ready!")

def load_sentiment_data(filename):
	sentiment_data = []
	with open(filename, 'r') as f:
		for line in f:
			_, sentiments = line.split(SEPARATOR)
			sentiment_data.append(eval(sentiments))
	return np.array(sentiment_data)			

def train_som(k, sentiment_data):
	som = MiniSom(k, k, sentiment_data.shape[1])
	print("Training SOM...", end="\r")
	som.train_random(sentiment_data, 100)
	print("Training complete!")
	return som

def plot_clusters(som, assignments):
	pass

def make_som(k, filter, n_segments):
	sentiments_file = SENTIMENT_OUTPUT_PATH.format(n_segments, filter)
	sentiment_data = load_sentiment_data(sentiments_file)
	som = train_som(k, sentiment_data)

if __name__ == "__main__":
	p = argparse.ArgumentParser(description='Generate SOM visualizations')

	p.add_argument("-k", "--k", type=int, required=True)
	p.add_argument("-f", "--filter", type=str, choices=['plain', 'savgol', 'slide'], required=True)
	p.add_argument("-s", "--n_segments", 
				   type=int, 
				   choices=[4, 8, 10, 12, 16, 20, 30, 40, 60, 120, 180, 240],
				   required=True)

	args = p.parse_args()

	make_som(k = args.k,
			 filter = args.filter,
			 n_segments = args.n_segments)