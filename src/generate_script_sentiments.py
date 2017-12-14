from constants import *

import numpy as np
from scipy.signal import savgol_filter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

import collections
import argparse
import math
import os

ANALYZER = SentimentIntensityAnalyzer()

def make_segments(script_lines):
    words = []
    for line in script_lines:
        words.extend(line.split())

    segment_length = int(len(words) / n_segments)
    if segment_length == 0: return None

    curr_start, curr_end = 0, segment_length
    segments = []
    while curr_end <= len(words):
        segment = words[curr_start:curr_end]
        segments.append(segment)
        curr_start += segment_length
        curr_end += segment_length

    return segments

def sentimentalize_script(script_lines, n_segments, filters):
    segments = make_segments(script_lines)
    if not segments:
        return None
    plain_sentiments = [ ANALYZER.polarity_scores(' '.join(segment))['compound'] for segment in segments ]

    sentiments = {}
    if 'plain' in filters:
        sentiments['plain'] = plain_sentiments
    if 'savgol' in filters:
        window_sz = int(math.sqrt(n_segments))
        if not window_sz % 2:
            window_sz += 1
        savgol_sentiments = list(savgol_filter(plain_sentiments, window_sz, 1))
        sentiments['savgol'] = savgol_sentiments
    if 'slide' in filters:
        window_sz = int(n_segments / 4)
        slide_sentiments = list(np.convolve(plain_sentiments, np.ones((window_sz,)) / window_sz, mode='same'))
        sentiments['slide'] = slide_sentiments

    return sentiments

def read_movie_lines():
    print ("Reading movie lines...")
    result = collections.defaultdict(list)
    with open(LINES_FILENAME, 'r', encoding=FILE_ENCODING) as f:
        for _line in tqdm([line for line in f]):
            tokens = _line.strip().split(SEPARATOR)
            if len(tokens) != 5: continue
            line_id, char_id, m_id, char_name, line = tokens
            line_id = int(line_id[1:])
            result[m_id].append((line_id, line))
    print ("Done reading movie lines.")

    print ("Sorting movie lines...")
    for m_id, m_lines in result.items():
        result[m_id] = [ line for line_id, line in sorted(m_lines) ]
    print ("Done sorting movie lines.")

    return result

def get_script_sentiments(movie_lines, n_segments, filters):
    result = collections.defaultdict(dict)
    n_lines = len(movie_lines)

    print ("Analyzing sentiment for segment size {}...".format(n_segments))
    for m_id, m_lines in tqdm(movie_lines.items()):
        sentiments = sentimentalize_script(m_lines, n_segments, filters)
        for filter_type in filters:
            if sentiments and sentiments[filter_type]:
                result[filter_type][m_id] = sentiments[filter_type]
    print ("Done analyzing sentiment for segment size {}.".format(n_segments))
    return result

def output_to_file(result, n_segments, filters):
    if not os.path.isdir(SENTIMENT_DIR_PATH.format(n_segments)):
        os.mkdir(SENTIMENT_DIR_PATH.format(n_segments))
    for filter_type in filters:
        print ("Outputting {} sentiments to file...".format(filter_type))
        with open(SENTIMENT_OUTPUT_PATH.format(n_segments, filter_type), 'w', encoding=FILE_ENCODING) as f:
            for m_id, sentiments in result[filter_type].items():
                f.write("{} {} {}\n".format(m_id, SEPARATOR, sentiments))
            f.flush()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-m', '--movie_id', type=str, default='all')
    parser.add_argument('-s', '--segment_nums', type=int, nargs='+', default=[4, 8, 10, 12, 16, 20, 30, 40, 60, 120, 180, 240])
    parser.add_argument('-f', '--filters', type=str, nargs='+', default=['plain', 'savgol', 'slide'])

    args = parser.parse_args()
    for filter_type in args.filters:
        if filter_type not in ['plain', 'savgol', 'slide']:
            print ("ERROR: invalid finter type \'{}\'".format(args.filter))
            sys.exit(1)

    movie_lines = read_movie_lines()
    for n_segments in args.segment_nums:
        result = get_script_sentiments(movie_lines, n_segments, args.filters)
        output_to_file(result, n_segments, args.filters)
