from collections import defaultdict
from constants import *
# TODO Document
class FeatureExtractor:

	def __init__(self, m_id, encoding):
		self.m_id = m_id
		self.feature_vectors = defaultdict(lambda : defaultdict(int))
		self.encoding = encoding

	def extract_features(self):
		raise NotImplementedError("extract_features method must be implemented in subclass")

	def get_feature_vector(self, x_id):
		return self.feature_vectors[x_id]


class ProtagonistFeatureExtractor(FeatureExtractor):

	def extract_features(self):
		self.get_character_counts()
		self.num_characters_talked_to()
		self.character_metadata()

		for k, v in self.feature_vectors.items():
			print("{} : {}".format(k, dict(v)))
			
	def get_character_counts(self):
		with open(MOVIE_LINES_FILENAME, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				char_id, movie_id, dialog = line[1], line[2], line[4]		
				if movie_id != self.m_id: continue
				self.feature_vectors[char_id]['line_count'] += 1
				self.feature_vectors[char_id]['word_count'] += len(dialog)

	def num_characters_talked_to(self):
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