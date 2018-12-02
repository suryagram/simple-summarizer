import argparse

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from string import punctuation

from heapq import nlargest
from collections import defaultdict


def main():
	args = parse_arguments()
	

	content = read_file(args.filepath)
	content = sanitize_input(content)
	
	sent_tokens, word_tokens = tokenize_content(content)

	sentence_rankings = score_tokens(word_tokens, sent_tokens)

	print(summarize(sentence_rankings, sent_tokens, args.length))


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('filepath', help="File Name of text doc to summarize.")
	parser.add_argument('-l', '--length', default=4, help="Number of sentences to return")
	args = parser.parse_args()

	return args


def read_file(fp):
	try:
		with open(fp, 'r') as file:
			return file.read()
	except IOError as e:
		print("Fatal Error, File at {} could not be located or is not readable.".format(fp))


def sanitize_input(data):
	replace = {
		ord('\f') : ' ',
		ord('\t') : ' ',
		ord('\n') : ' ',
		ord('\r') : None
	}

	return data.translate(replace)


def tokenize_content(data):
	stop_words = list(set(stopwords.words('english'))) + list(punctuation)
	words = word_tokenize(data)

	return [
		sent_tokenize(data),
		[word for word in words if word not in stop_words]
		]


def score_tokens(filtered_words, sent_tokens):

	word_freq = FreqDist(filtered_words)

	ranking = defaultdict(int)

	for i, sentence in enumerate(sent_tokens):
		for word in word_tokenize(sentence.lower()):
			if word in word_freq:
				ranking[i] += word_freq[word]


	return ranking


def summarize(rankings, sentences, length):
	if int(length) > len(sentences):
		print("Error, number of total sentences in is less than number of sentences requested for summary. Use --l flag to adjust")
		exit()

	indexes = nlargest(int(length), rankings, key = rankings.get)
	final_sentences = [sentences[i] for i in sorted(indexes)]

	return ' '.join(final_sentences)






if __name__ ==  "__main__":
	print((main()))

