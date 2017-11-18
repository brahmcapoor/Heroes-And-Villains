from constants import *
from FeatureExtractors import ProtagonistFeatureExtractor, AntagonistFeatureExtractor

"""
Currently just an example of how to use FeatureExtractors
"""

def print_feature_vectors(movie_id):
	extractor = AntagonistFeatureExtractor(movie_id, FILE_ENCODING, 'u362')
	extractor.extract_features()
	for char in extractor.characters():
		print("{}:{}".format(char, extractor[char]))

if __name__ == "__main__":
	movie_id = input("Movie id (format m<number>): ")
	print_feature_vectors(movie_id)