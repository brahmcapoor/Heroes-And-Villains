import numpy as np
from constants import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import FeatureExtractors
from FeatureExtractors import ProtagonistFeatureExtractor, AntagonistFeatureExtractor

"""
Currently just an example of how to use FeatureExtractors
"""

def print_feature_vectors(movie_id):
	extractor = AntagonistFeatureExtractor(movie_id, FILE_ENCODING, 'u362')
	extractor.extract_features()
	for char in extractor.characters():
		print("{}:{}".format(char, FeatureExtractors.get_nparray(extractor, char)))

def generate_data_and_labels():

	X = []
	y = []
	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		for movie in dataset:
			movie = movie.split(SEPARATOR)
			m_id, protagonist = movie[0:2]
			extractor = ProtagonistFeatureExtractor(m_id, FILE_ENCODING)
			extractor.extract_features()
			for char in extractor.characters():
				feature_vector = FeatureExtractors.get_nparray(extractor, char)
				X.append(feature_vector)
				y.append(int(char == protagonist))

	X = np.array(X).T
	y = np.array(y).T
	return np.array(X).T, np.array(y).T

def train_logistic_regression(train_x, train_y):
	model = LogisticRegression()
	model.fit(train_x, train_y)
	return model

def model_accuracy(model, features, targets):
	return model.score(features, targets)

if __name__ == "__main__":

	X, y = generate_data_and_labels()

	train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.75)
	model = train_logistic_regression(train_x, train_y)

	train_accuracy = model_accuracy(model, train_x, train_y)
	test_accuracy = model_accuracy(model, test_x, test_y)

	print("Train accuracy: {}".format(train_accuracy))
	print("Test accuracy: {}".format(test_accuracy))
