SEPARATOR =  " +++$+++ "

FILE_ENCODING =  "iso-8859-1"

DATASET_PATH = "../dataset"
MOVIE_METADATA_FILENAME = "{}/movie_titles_metadata.txt".format(DATASET_PATH)
MOVIE_LINES_FILENAME = "{}/movie_lines.txt".format(DATASET_PATH)
CHARACTER_METADATA_FILENAME = "{}/movie_characters_metadata.txt".format(DATASET_PATH)
MOVIE_CONVERSATIONS_FILENAME = "{}/movie_conversations.txt".format(DATASET_PATH)
LABELLED_DATA_FILENAME = "{}/movie_pa_labels.txt".format(DATASET_PATH)
SCRIPT_URL_FILENAME = "{}/raw_script_urls.txt".format(DATASET_PATH)
LINES_FILENAME = "{}/movie_lines.txt".format(DATASET_PATH)

SENTIMENT_DIR_PATH = '../sentiments/{}_segments'
SENTIMENT_OUTPUT_PATH = '../sentiments/{}_segments/{}_sentiments.txt'
CLUSTER_DIR_PATH = '../plots/{}_segments/k_{}/filter_{}/'
CLUSTER_OUTPUT_CENTROID_PATH = '../plots/{}_segments/k_{}/filter_{}/centroids.txt'
CLUSTER_OUTPUT_ASSIGNMENT_PATH = '../plots/{}_segments/k_{}/filter_{}/assignments.txt'
SEGMENT_SIZES = [4, 8, 10, 12, 16, 20, 30, 40, 60, 120, 240]
