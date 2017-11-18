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

def generate_data_and_labels(protagonist, protagonist_map=None):
	print("Generating data for {}...".format("protagonist" if protagonist else "antagonist"))
	X = []
	y = []

	char_idx = 1 if protagonist else 2
	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		for movie in dataset:
			movie = movie.strip().split(SEPARATOR)
			m_id, character = movie[0], movie[char_idx]
			extractor = None
			if character:
				extractor = ProtagonistFeatureExtractor(m_id, FILE_ENCODING)
			else:
				extractor = AntagonistFeatureExtractor(m_id, FILE_ENCODING, protagonist_map[m_id])
			extractor.extract_features()
			for char in extractor.characters():
				feature_vector = FeatureExtractors.get_nparray(extractor, char)
				X.append(feature_vector)
				y.append(int(char == character))

	# print(any(x != 0 for x in y))
	return np.array(X), np.array(y)

def train_logistic_regression(train_x, train_y):
	model = LogisticRegression()
	model.fit(train_x, train_y)
	return model

def model_accuracy(model, features, targets):
	return model.score(features, targets)

def find_movie_protagonist(model, m_id):
	extractor = ProtagonistFeatureExtractor(m_id, FILE_ENCODING)
	extractor.extract_features()
	curr_protag = None
	max_prob = -float('inf')
	for char in extractor.characters():
		feature_vector = FeatureExtractors.get_nparray(extractor, char)
		prob = model.predict_proba(np.array([feature_vector]))
		if prob[0,1] > max_prob:
			max_prob = prob[0,1]
			curr_protag = char
	return curr_protag

def generate_protagonist_map(model):
	print("Generating protagonist map...")
	protagonists = {}
	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		for movie in dataset:
			movie = movie.split(SEPARATOR)
			m_id = movie[0]
			protagonists[m_id] = find_movie_protagonist(model, m_id)
	return protagonists


if __name__ == "__main__":

	protagonist_X, protagonist_Y = generate_data_and_labels(True)

	p_train_x, p_test_x, p_train_y, p_test_y = train_test_split(protagonist_X, protagonist_Y, train_size=0.75)
	print("Training protagonist model...")
	protagonist_model = train_logistic_regression(p_train_x, p_train_y)

	p_train_accuracy = model_accuracy(protagonist_model, p_train_x, p_train_y)
	p_test_accuracy = model_accuracy(protagonist_model, p_test_x, p_test_y)

	print("Protagonist train accuracy: {}".format(p_train_accuracy))
	print("Protagonist test accuracy: {}".format(p_test_accuracy))

	protagonists = generate_protagonist_map(protagonist_model)

	antagonist_X, antagonist_Y = generate_data_and_labels(False, protagonists)

	print("Training anatagonist model...")
	a_train_x, a_test_x, a_train_y, a_test_y = train_test_split(antagonist_X, antagonist_Y, train_size=0.75)
	antagonist_model = train_logistic_regression(a_train_x, a_train_y)

	a_train_accuracy = model_accuracy(antagonist_model, a_train_x, a_train_y)
	a_test_accuracy = model_accuracy(antagonist_model, a_test_x, a_test_y)

	print("Antagonist train accuracy: {}".format(a_train_accuracy))
	print("Antagonist test accuracy: {}".format(a_test_accuracy))


