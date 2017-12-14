from constants import *

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

import argparse
import sys

def read_sentiment_files(n, filter_type):
    result = {}
    with open(SENTIMENT_OUTPUT_PATH.format(n, filter_type), 'r', encoding=FILE_ENCODING) as f:
        for line in f:
            tokens = line.split(SEPARATOR)
            sentiments = eval(tokens[1])
            result[tokens[0].strip()] = sentiments[0:n]
    return result

def graph_sentiment(sentiments, label):
    plt.gcf().clear()
    plt.xlabel('Segment of Screenplay')
    plt.ylabel('Average Dialogue Sentiment')
    plt.title(label)

    axes = plt.gca()
    axes.set_ylim([-1.5,1.5])

    plt.plot(sentiments, color="red")
    plt.scatter(range(len(sentiments)), sentiments, marker="o", c='r')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-m', '--movie_id', type=str)
    parser.add_argument('-n', '--n_segments', type=int)
    parser.add_argument('-f', '--filter', type=str)
    parser.add_argument('-g', '--graph', type=str)
    parser.add_argument('-l', '--label', type=str)

    args = parser.parse_args()
    if args.filter not in ['plain', 'savgol', 'slide']:
        print ("ERROR: invalid filter type \'{}\'".format(args.filter))
        sys.exit(1)

    if args.graph not in ['sentiment']:
        print ("ERROR: invalid graph type \'{}\'".format(args.graph))
        sys.exit(1)

    sentiments = read_sentiment_files(args.n_segments, args.filter)
    if args.movie_id not in sentiments:
        print ("ERROR: no sentiment data for movie id \'{}\' ({} segments, {} filter)".format(args.movie_id, args.n_segments, args.filter))
        sys.exit(1)

    if args.graph == 'sentiment':
        graph_sentiment(sentiments[args.movie_id], args.label)
