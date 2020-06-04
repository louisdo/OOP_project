import math
import json
import os
import logging
from collections import Counter
from .utils import Utils


class TFIDF:
    """
    An utility to compute term frequency - inverse document frequency score
    """
    def __init__(self, 
                 corpus: "list of lists",
                 vocab_save_path: str = None):
        """
        input:
            + corpus: a list of lists containing tokenized terms from the processed data
            + vocab_save_path: if provided, save vocab to specified path,
            vocab contains 2 files, 'index2word.json' and 'word2index.json'
        """
        self.corpus = corpus

        self.vocab = self._get_vocab(save_path=vocab_save_path)

    def _tf(self, sentence: list) -> dict:
        """
        Compute term frequency score

        input:
            + sentence: a list of tokenized terms
        output:
            + a dictionary with its keys are the terms in the 
            sentence, each comes with a corresponding value showing
            its tf score
        """

        # calculate term frequency
        term_counts = Counter()
        for term in sentence:
            term_counts[term] += 1

        term_counts = list(term_counts.items())

        max_term_count = max(term_counts, key = lambda x: x[1])[1]

        tf_scores = {}

        # get tf score for each term
        for index in range(len(term_counts)):
            term = term_counts[index][0]
            count = term_counts[index][1]

            tf_score = 0.5 + 0.5 * (count / max_term_count)

            tf_scores[term] = tf_score

        return tf_scores

    def _get_vocab(self, save_path = None) -> dict:
        """
        Get vocabulary from corpus

        input: 
            + save_path: if provided, save 'word2index.json' and 'index2word.json' to specified path
        output:
            + a dictionary with its keys are the terms in the vocab, each comes with a corresponding values
            specify its frequency in the vocab
        """
        vocab_count = Counter()

        # get vocab frequency for every terms
        for sentence in self.corpus:
            sentence = list(set(sentence))
            for term in sentence:
                vocab_count[term] += 1

        # get word2index and index2word, which are 2 components of the vocab
        vocab_words = list(vocab_count.keys())
        index2word = {index+1:vocab_words[index] for index in range(len(vocab_words))}
        word2index = {vocab_words[index]:index+1 for index in range(len(vocab_words))}
        index2word[0] = "<pad>"
        word2index["<pad>"] = 0
        index2word[len(vocab_words) + 1] = "<sos>"
        word2index["<sos>"] = len(vocab_words) + 1
        index2word[len(vocab_words) + 2] = "<eos>"
        word2index["<eos>"] = len(vocab_words) + 2

        # if save_path is provided, save 'word2index.json' and 'index2word.json' to specified path
        if save_path is not None:
            Utils.save_vocab(vocab_folder = save_path,
                             index2word = index2word,
                             word2index = word2index)

            logging.info("Saved 'index2word.json' and 'word2index.json' to {}".format(save_path))

        return vocab_count


    def _idf(self, sentence: list) -> dict:
        """
        Compute inverse document frequecy score

        input:
            + sentence: a list of tokenized terms
        output:
            + a dictionary with its keys are the terms in the 
            sentence, each comes with a corresponding value showing
            its idf score
        """
        idf_scores = {}

        for term in sentence:
            try:
                idf_score = math.log(len(self.corpus) / self.vocab[term])
                idf_scores[term] = idf_score
            except Exception:
                continue

        return idf_scores

    def tfidf_score(self, sentence: list):
        """
        Compute tfidf score

        input:
            + sentence: a list of tokenized terms
        output:
            + a dictionary with its keys are the terms in the 
            sentence, each comes with a corresponding value showing
            its tfidf score
        """
        tf_scores = self._tf(sentence)
        idf_scores = self._idf(sentence)

        tfidf_scores = []

        for index in range(len(sentence)):
            term = sentence[index]
            tf_score = tf_scores[term]
            idf_score = idf_scores[term]

            tfidf_scores.append((term, tf_score * idf_score))

        return tfidf_scores