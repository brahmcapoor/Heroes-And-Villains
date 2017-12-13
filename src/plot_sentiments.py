from constants import *
import requests as r
from bs4 import BeautifulSoup as b
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math
import collections
from scipy.signal import savgol_filter


def fetch_script_text(script_url):
    script = r.get(script_url).text
    clean = b(script, "html.parser").text
    if len(clean) < 10000: return None
    return clean

def get_lines(script_text):
    for line in script_text.split('\n'):
        if line.isupper() and line.isalpha():
            print (line)

def script_sentiment_summary(script_lines, n_segments):
    analyzer = sia()

    words = []
    for line in script_lines:
        words.extend(line.split())

    segment_length = int(len(words) / n_segments)
    if segment_length == 0: return None, None

    curr_start = 0
    curr_end = segment_length
    segments = []
    while curr_end <= len(words):
        segment = words[curr_start:curr_end]
        segments.append(segment)
        curr_start += segment_length
        curr_end += segment_length

    lengths = [len(segment) for segment in segments]
    sentiments = [ analyzer.polarity_scores(' '.join(segment))['compound'] for segment in segments ]

    window_sz = round(math.sqrt(n_segments))
    window_sz = window_sz if window_sz % 2 else window_sz + 1
    smoothed = list(savgol_filter(sentiments, window_sz, 1))

    return sentiments, smoothed


def read_movie_lines():
    result = collections.defaultdict(list)
    with open(LINES_FILENAME, 'r', encoding=FILE_ENCODING) as f:
        for _line in f:
            tokens = _line.strip().split(SEPARATOR)
            if len(tokens) != 5: continue
            line_id, char_id, m_id, char_name, line = tokens
            line_id = int(line_id[1:])
            result[m_id].append((line_id, line))

    for m_id, m_lines in result.items():
        result[m_id] = [ line for line_id, line in sorted(m_lines) ]

    return result

def get_script_sentiments(movie_lines, n_segments):
    result = []
    n_lines = len(movie_lines)

    print ("\rAnalyzing sentiment for segment size {}...".format(n_segments), end="")
    for index, m_info in enumerate(movie_lines.items()):
        m_id, m_lines = m_info
        print ("\rAnalyzing sentiment for segment size {}... {:5.2f}% done...".format(n_segments, index / float(n_lines) * 100), end="")
        summary, smoothed = script_sentiment_summary(m_lines, n_segments)
        if summary: result.append((m_id, summary, smoothed))
    print ("\rAnalyzing sentiment for segment size {}... Done!           ".format(n_segments))

    return result

def output_to_file(result, n_segments):
    with open('../sentiments/sent_{}.txt'.format(n_segments), 'w', encoding=FILE_ENCODING) as plain, open('../sentiments/sent_{}_savgol.txt'.format(n_segments), 'w', encoding=FILE_ENCODING) as smooth:
        for m_id, summary, smoothed in result:
            plain.write("{} {} {}\n".format(m_id, SEPARATOR, summary))
            smooth.write("{} {} {}\n".format(m_id, SEPARATOR, smoothed))
        plain.flush()
        smooth.flush()


def main():
    print ("Reading movie lines... ", end="")
    movie_lines = read_movie_lines()
    print ("Done!")

    segment_nums = [4, 8, 10, 12, 16, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400]
    for n_segments in segment_nums:
        result = get_script_sentiments(movie_lines, n_segments)
        output_to_file(result, n_segments)

if __name__ == '__main__':
    main()
