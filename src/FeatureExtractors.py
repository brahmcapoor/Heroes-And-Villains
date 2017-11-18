import ast
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
from constants import *

class FeatureExtractor:
	"""
	Base class for all feature extractors. Exports methods to 
	get all characters with a feature extractor as well as to 
	retrieve the feature vector associated with a particular vector.

	IMPORTANT: The base class also has an abstract method `extract_features` 
	which MUST be overridden by subclasses representing specific extractors. 
	The base class exists solely to provide `characters()` and the [] operator.
	"""
	def __init__(self, m_id, encoding):
		"""
		Feature vector maps character ids to feature vectors.
		Feature vectors are sparse vectors represented by default dicts. 
		"""
		self.m_id = m_id
		self.feature_vectors = defaultdict(lambda : defaultdict(float))
		self.encoding = encoding

	def extract_features(self):
		"""
		Abstract method. Must be overwritten in subclasses.
		"""
		raise NotImplementedError("extract_features method must be implemented in subclass")

	def characters(self):
		"""
		Returns characters which have feature vectors
		"""
		return self.feature_vectors.keys()

	def __getitem__(self, key):
		"""
		Access the feature vector of a particular character
		"""
		return self.feature_vectors[key]

class ProtagonistFeatureExtractor(FeatureExtractor):
	"""
	Feature extractor for protagonist identification
	"""

	def extract_features(self):
		"""
		Overrides base class method
		"""
		self.feature_names = ["line_count", "word_count", "num_spoken_to", "in_title", "credits_pos",
							  "is_male", "is_female"]
		self.get_character_counts()
		self.num_characters_talked_to()
		self.character_metadata()
			
	def get_character_counts(self):
		"""
		Get character word counts and line counts
		"""
		with open(MOVIE_LINES_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				char_id, movie_id, dialog = line[1], line[2], line[4]		
				if movie_id != self.m_id: continue
				self.feature_vectors[char_id]['line_count'] += 1
				self.feature_vectors[char_id]['word_count'] += len(dialog)

	def num_characters_talked_to(self):
		"""
		Number of characters that the character has a conversation with
		"""
		chars_spoken_to = defaultdict(set)
		with open(MOVIE_CONVERSATIONS_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				char_1, char_2, movie_id = line[0:3]
				if movie_id != self.m_id: 
					continue
				chars_spoken_to[char_1].add(char_2)
				chars_spoken_to[char_2].add(char_1)

		for char in chars_spoken_to:
			self.feature_vectors[char]['num_spoken_to'] = len(chars_spoken_to[char])

	def character_metadata(self):

		"""
		Name in title, position in credits, gender
		"""

		def name_in_title(char_name, movie_name):
			return any(token.lower() in movie_name.lower() for token in char_name.split())

		with open(CHARACTER_METADATA_FILENAME, 'r', encoding=self.encoding) as f:

			for line in f:
				char_id, char_name, movie_id, movie_name, gender, credits_pos = line.strip().split(SEPARATOR)

				if movie_id != self.m_id:
					continue

				if name_in_title(char_name, movie_name):
					self.feature_vectors[char_id]['in_title'] = 1

				if credits_pos != "?":
					self.feature_vectors[char_id]['credits_pos'] = int(credits_pos)

				if gender == "m":
					# separated from 'f' in case character isn't human
					self.feature_vectors[char_id]['is_male'] = 1 

				if gender == "f":
					self.feature_vectors[char_id]['is_female'] = 1

class AntagonistFeatureExtractor(ProtagonistFeatureExtractor):

	def __init__(self, m_id, file_encoding, protagonist_id):
		super(AntagonistFeatureExtractor, self).__init__(m_id, file_encoding)
		self.analyzer = SentimentIntensityAnalyzer()
		self.protagonist = protagonist_id

	def extract_features(self):
		super(AntagonistFeatureExtractor, self).extract_features()
		self.feature_names += ["average_sentiment", "protagonist_sentiment"]
		self.get_average_sentiment()
		self.protagonist_sentiment()
		# self.num_mentioned()

	def get_average_sentiment(self):
		sentiments = defaultdict(lambda : [0.0 , 0])
		with open(MOVIE_LINES_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				char_id, movie_id, dialog = line[1], line[2], line[4]
				if movie_id != self.m_id: 
					continue
				sentiment = self.analyzer.polarity_scores(dialog)['compound']
				sentiments[char_id][0] += sentiment
				sentiments[char_id][1] += 1

		for char_id in sentiments:
			self.feature_vectors[char_id]['average_sentiment'] = sentiments[char_id][0]/sentiments[char_id][1]

	def protagonist_sentiment(self):
		line_sentiments = {}
		with open(MOVIE_LINES_FILENAME, 'r', encoding = self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				line_id, char_id, movie_id, dialog = line[0], line[1], line[2], line[4]
				if movie_id != self.m_id: 
					continue
				if char_id == self.protagonist:
					line_sentiments[line_id] = self.analyzer.polarity_scores(dialog)['compound']

		protagonist_sentiments = defaultdict(lambda: [0.0 ,0])

		with open(MOVIE_CONVERSATIONS_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				char_1, char_2, movie_id, lines = line.split(SEPARATOR)
				lines = ast.literal_eval(lines)
				if movie_id != self.m_id:
					continue
				if self.protagonist in [char_1, char_2]:
					for l in lines:
						if l in line_sentiments:
							if char_1 == self.protagonist:
								protagonist_sentiments[char_2][0] += line_sentiments[l]
								protagonist_sentiments[char_2][1] += 1
								self.feature_vectors[char_2]['num_p_interactions'] += len(lines)
							else:
								protagonist_sentiments[char_1][0] += line_sentiments[l]
								protagonist_sentiments[char_1][1] += 1
								self.feature_vectors[char_1]['num_p_interactions'] += len(lines)

		for char_id in protagonist_sentiments:
			self.feature_vectors[char_id]['protagonist_sentiment'] = protagonist_sentiments[char_id][0]/protagonist_sentiments[char_id][1]

	def num_mentioned(self):

		def name_included(name, dialog):
			return any(token.lower() in dialog.lower().split() for token in name.split())

		ids_to_names = {}
		with open(CHARACTER_METADATA_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				char_id, name, movie_id = line[0:3]
				if movie_id != self.m_id:
					continue
				ids_to_names[char_id] = name

		with open(MOVIE_LINES_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				movie_id, dialog = line[2], line[4]
				for char_id in self.feature_vectors:
					if name_included(ids_to_names[char_id], dialog):
						self.feature_vectors[char_id]['num_mentioned'] += 1

def get_nparray(extractor, x_id):

	try:
		feature_names = extractor.feature_names
	except (AttributeError):
		raise AttributeError("You need to extract features first")

	feature_vector = extractor[x_id]
	array = [feature_vector[feature_name] for feature_name in feature_names]
	return np.array(array, dtype=np.float32)