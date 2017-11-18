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
		self.feature_vectors = defaultdict(lambda : defaultdict(int))
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

		with open(MOVIE_CHARACTER_METADATA_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				char_id, char_name, movie_id, movie_name, gender, credits_pos = line.strip().split(SEPARATOR)
				if movie_id != self.m_id:
					continue

				if name_in_title(char_name, movie_name):
					self.feature_vectors[char_id]['in_title'] = 1

				if credits_pos != "?":
					self.feature_vectors[char_id]['credits_pos'] = int(credits_pos)

				if gender == "m":
					self.feature_vectors[char_id]['is_male'] = 1 #separated from 'f' in case character isn't human

				if gender == "f":
					self.feature_vectors[char_id]['is_female'] = 1