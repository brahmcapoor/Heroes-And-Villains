from collections import defaultdict

SEPARATOR =  " +++$+++ "
FILE_ENCODING =  "iso-8859-1"

MOVIE_METADATA_FILENAME = "../../dataset/movie_titles_metadata.txt"
MOVIE_LINES_FILENAME = "../../dataset/movie_lines.txt"

class Baseline():

	def __init__(self, movie_ids_file, movie_lines_file, file_encoding):
		self.encoding = file_encoding
		self.get_movie_ids(movie_ids_file)
		self.get_line_frequencies(movie_lines_file)
		self.find_protagonists_and_antagonists()

	def get_movie_ids(self, filename):
		self.ids = {}

		with open(filename, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				movie_id, movie_name = line[0], line[1]
				self.ids[movie_id] = movie_name


	def get_line_frequencies(self, filename):
		self.frequencies = defaultdict(lambda: defaultdict(int))

		with open(filename, 'r', encoding=self.encoding) as f:
			for line in f:
				line = line.split(SEPARATOR)
				movie_id, character = line[2], line[3]
				self.frequencies[movie_id][character] += 1

	def find_protagonists_and_antagonists(self):
		self.answers = {}
		for movie_id, movie in self.frequencies.items():
			freq_table = list(movie.items())
			freq_table.sort(key = lambda x: x[1], reverse=True)
			self.answers[movie_id] = (freq_table[0][0], freq_table[1][0])

		self.movies_to_ids = {title : movie_id for movie_id, title in self.ids.items()}

	def print_movie_answer(self, movie, by_name=False):
		if by_name:	
			if movie.lower() not in self.movies_to_ids:
				print("Movie not in dataset")
				return

			movie = self.movies_to_ids[movie]
		
		answer = self.answers[movie]
		movie_name = self.ids[movie]
		print("In {}, the protagonist is {} and the antagonist is {}".format(
			movie_name, answer[0], answer[1]))	

	def simulate(self):
		while True:
			query = input("Type name of movie for specific answer, * for all movies and blank to quit: ")
			if query == "*":
				for movie in self.ids:
					self.print_movie_answer(movie)
			elif query != "":
				if query in self.ids:
					self.print_movie_answer(query)
				else:
					self.print_movie_answer(query, by_name=True)
			elif query == "":
				return
			else:
				print("Invalid input")
			print("")



if __name__ == '__main__':
	print("Running baseline")

	baseline = Baseline(movie_ids_file = MOVIE_METADATA_FILENAME, 
						movie_lines_file = MOVIE_LINES_FILENAME, 
						file_encoding = FILE_ENCODING)

	baseline.simulate()