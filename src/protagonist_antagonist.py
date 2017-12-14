from constants import *
from FeatureExtractors import *

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import dill

import collections
import argparse

def generate_data_and_labels(protagonist, protagonist_map=None):
	print("Generating data for {}...".format("protagonist" if protagonist else "antagonist"))

	data = collections.defaultdict(dict)

	char_idx = 1 if protagonist else 2
	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		average_sentiments = {}
		protag_sentiments = {}

		for movie in dataset:
			movie = movie.strip().split(SEPARATOR)
			m_id, character = movie[0], movie[char_idx]
			extractor = None
			if protagonist:
				extractor = ProtagonistFeatureExtractor(m_id, FILE_ENCODING)
			else:
				extractor = AntagonistFeatureExtractor(m_id, FILE_ENCODING, protagonist_map[m_id])
			extractor.extract_features()
			
			if not protagonist:
				average_sentiments[m_id] = extractor.avg_sentiments
				protag_sentiments[m_id] = extractor.protag_sentiments

			# add extractor.protagonist_sentiments to protag_sentiments
			for char in extractor.characters():
				feature_vector = get_nparray(extractor, char)
				data[m_id][char] = (feature_vector, int(char == character))

		if not protagonist:
			with open('avg_sentiment.pkl', 'wb') as f:
				dill.dump(average_sentiments, f)
				f.close()

			with open('protagonist_sentiment.pkl', 'wb') as f:
				dill.dump(protag_sentiments, f)
				f.close()

	return data

def train_logistic_regression(train_x, train_y):
	model = LogisticRegression()
	model.fit(train_x, train_y)
	return model

def model_accuracy(model, m_ids, m_ids_to_ids, protagonist_map=None):
	total = 0.0
	misclassifications = []
	for m_id in m_ids:
		if not protagonist_map:
			predicted = find_movie_agonist(model, m_id)
		else:
			predicted = find_movie_agonist(model, m_id, protagonist_id=protagonist_map[m_id])
		if predicted == m_ids_to_ids[m_id]:
			total += 1
		else:
			misclassifications.append((m_id, predicted, m_ids_to_ids[m_id]))
	return total / len(m_ids), misclassifications

def find_movie_agonist(model, m_id, protagonist_id=None):
	if not protagonist_id:
		extractor = ProtagonistFeatureExtractor(m_id, FILE_ENCODING)
	else:
		extractor = AntagonistFeatureExtractor(m_id, FILE_ENCODING, protagonist_id)
	extractor.extract_features()
	curr_tag = None
	max_prob = -float('inf')
	for char in extractor.characters():
		feature_vector = get_nparray(extractor, char)
		prob = model.predict_proba(np.array([feature_vector]))
		if prob[0,1] > max_prob:
			max_prob = prob[0,1]
			curr_tag = char
	return curr_tag

def generate_protagonist_map(model):
	print("Generating protagonist map...")
	protagonists = {}
	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		for movie in dataset:
			movie = movie.split(SEPARATOR)
			m_id = movie[0]
			protagonists[m_id] = find_movie_agonist(model, m_id)
	return protagonists

def assemble_data(data, train_set, test_set):
	train_x = []
	train_y = []
	train_m_ids_to_ids = {}
	test_m_ids_to_ids = {}

	for m_id in train_set:
		for char_example, info in data[m_id].items():
			train_x.append(info[0])
			train_y.append(info[1])
			if info[1] == 1:
				train_m_ids_to_ids[m_id] = char_example
	
	for m_id in test_set:
		for char_example, info in data[m_id].items():
			if info[1] == 1:
				test_m_ids_to_ids[m_id] = char_example

	return train_x, train_y, train_m_ids_to_ids, test_m_ids_to_ids

def print_misclassifications(train_misclassifications, test_misclassifications, protagonist):
	print ("Misclassifications for {} Model:".format('Protagonist' if protagonist else 'Antagonist'))
	
	print ("\tTrain:")
	for misclassification in train_misclassifications:
		print ("\t - [{}] Predicted: {}, Actual: {}".format(*misclassification))
	print ("\tTest:")
	for misclassification in test_misclassifications:
		print ("\t - [{}] Predicted: {}, Actual: {}".format(*misclassification))

if __name__ == "__main__":
	p = argparse.ArgumentParser(description='Generate SOM visualizations')
	p.add_argument("-p", "--print_errors", type=bool, default=False)
	args = p.parse_args()

	p_data = generate_data_and_labels(True)
	p_train_set, p_test_set = train_test_split(list(p_data.keys()), train_size=0.75)

	p_train_x, \
	p_train_y, \
	p_train_m_ids_to_p_ids, \
	p_test_m_ids_to_p_ids = assemble_data(p_data, p_train_set, p_test_set)

	print("Training protagonist model...")
	protagonist_model = train_logistic_regression(p_train_x, p_train_y)

	p_train_accuracy, p_train_misclassifications 	= model_accuracy(protagonist_model, p_train_set, p_train_m_ids_to_p_ids)
	p_test_accuracy, p_test_misclassifications 		= model_accuracy(protagonist_model, p_test_set, p_test_m_ids_to_p_ids)
	print("Protagonist train accuracy: {}".format(p_train_accuracy))
	print("Protagonist test accuracy: {}".format(p_test_accuracy))
	if args.print_errors:
		print_misclassifications(p_train_misclassifications, p_test_misclassifications, True)

	protagonists = generate_protagonist_map(protagonist_model)
	a_data = generate_data_and_labels(False, protagonist_map=protagonists)
	a_train_set, a_test_set = train_test_split(list(a_data.keys()), train_size=0.75)

	a_train_x, \
	a_train_y, \
	a_train_m_ids_to_a_ids, \
	a_test_m_ids_to_a_ids = assemble_data(a_data, a_train_set, a_test_set)

	print("Training anatagonist model...")
	antagonist_model = train_logistic_regression(a_train_x, a_train_y)

	a_train_accuracy, a_train_misclassifications 	= model_accuracy(antagonist_model, a_train_set, a_train_m_ids_to_a_ids, protagonist_map=protagonists)
	a_test_accuracy, a_test_misclassifications 		= model_accuracy(antagonist_model, a_test_set, a_test_m_ids_to_a_ids, protagonist_map=protagonists)
	print("Antagonist train accuracy: {}".format(a_train_accuracy))
	print("Antagonist test accuracy: {}".format(a_test_accuracy))
	if args.print_errors:
		print_misclassifications(a_train_misclassifications, a_test_misclassifications, True)


