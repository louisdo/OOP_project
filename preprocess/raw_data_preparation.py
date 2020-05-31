import re
import logging
import json
import os
import pandas as pd
from argparse import ArgumentParser
from pyvi import ViTokenizer
from tqdm import tqdm
from lib.tf_idf import TFIDF

tqdm.pandas()

class RawDataPrep:
    def __init__(self):
        pass


    @staticmethod
    def get_patterns():
        """
        Get pattern for word filtering
        """
        regexes = {
            "number": "[-+]?\d+([\.,]?\d+)*(%){0}(\s{1}(tỷ|triệu|chục|trăm|nghìn|ngàn){1,2}){0}",
            "date": "\(?([1-9]|[12][0-9]|3[01])+\s?[/-]{1}\s?([1-9]|1[0-2])+[[/-]?\d*]{0,1}\)?",
            "percent": "\(?[-+]?\d+[\.,]?\d*(%){1}\)?",
            "word_number": "[-+]?\d+([\.,]?\d+)*(%){0}(\s{1}(tỷ|triệu|chục|trăm|nghìn|ngàn)){1,2}",
            "time": "\d{1,2}h\d{1,2}"
        }

        patterns = {}

        for reg in regexes:
            patterns[reg] = re.compile(regexes[reg])

        return patterns


    @staticmethod
    def process_sentence(sentence, patterns):
        string = sentence.strip(".").replace(", ", " , ")
        char2replace = ["(", ")", "[", "]",
                        "{", "}", '""', "''",
                        "``", ]
        for character in char2replace:
            string = string.replace(character, "")
        string = string.lower()
        string = string.replace("vn-index", "vnindex")

        pattern_count = {pattern: 0 for pattern in patterns}

        replaced_terms = {}
        for pattern in ["date", "percent", "word_number", "time", "number"]:
            terms_found = [item[0] for item in list(patterns[pattern].finditer(string))]
            for index in range(len(terms_found)):
                replaced_terms[pattern + "_{}".format(index + 1)] = terms_found[index]
            string = patterns[pattern].sub(pattern, string)

        string = string.split(" ")

        for index in range(len(string)):
            term = string[index]
            if term in ["date", "percent", "word_number", "time", "number"]:
                pattern_count[term] += 1
                encoded_term = term + "_{}".format(pattern_count[term])
                string[index] = encoded_term

        string = " ".join(string)

        if pattern_count["number"] == 0 and pattern_count["percent"] == 0 \
            and pattern_count["word_number"] == 0:
           return None, None

        return string.lower(), replaced_terms

    @staticmethod
    def get_important_word(sentence, tfidf):
        scores = tfidf.tfidf_score(sentence)

        values = sorted([item[1] for item in scores])
        median = values[int(2 * len(values) / 3)]

        important_words = [item[0] for item in scores if item[1] >= median or \
                                                            item[0] == "vnindex" or \
                                                            item[0].startswith("number_") or \
                                                            item[0].startswith("date_") or \
                                                            item[0].startswith("percent_") or \
                                                            item[0].startswith("word_number_") or \
                                                            item[0].startswith("time_")]

        return "|".join(important_words)

    @staticmethod
    def encode_line(sentence, word2index):
        encoded_sentence = [word2index["<sos>"]] + [word2index[word] for word in sentence] + [word2index["<eos>"]]
        return [str(item) for item in encoded_sentence]


    def get_data_from_file(self, data_path: str) -> list:
        """
        input: 
            + data_path: path to txt file containing data
        output:
            + a list containing raw sentences
        """

        with open(data_path, encoding = "utf-8") as f:
            file_lines = f.readlines()

        line_regex = "[0-9]+[:]{1}.+"
        line_pattern = re.compile(line_regex)

        data = []
        for index in range(len(file_lines)):
            line = file_lines[index]
            if line_pattern.match(line):
                data.append(line)

        patterns = self.get_patterns()

        processed_data = []

        for index in tqdm(range(len(data)), "Getting data"):
            string = data[index].split(":")[1].strip()
            processed_string = self.process_sentence(string, patterns)[0]
            if processed_string is not None:
                processed_data.append(processed_string)

        return processed_data

        

    def get_corpus(self, data_path: str):
        """
        input:
            + data_path: path to txt file containing raw data
        output:
            + a list of list which is the corpus
        """
        corpus = []
        
        data = self.get_data_from_file(data_path)

        max_number_count = -1

        for sentence in tqdm(data, desc="Tokenizing data"):
            tokenized_sentence = ViTokenizer.tokenize(sentence)
            tokens = tokenized_sentence.split(" ")
            
            corpus.append(tokens)

        logging.info("Max number of numbers in a sentence is {}".format(max_number_count))

        return corpus

    def get_training_data(self, 
                          data_path: str, 
                          vocab_save_path: str):
        """
        input:
            + data_path: path to txt file containing raw data
            + vocab_save_path: path to where to 2 files 'index2word.json' and 
            'word2index.json' will be saved
        output:
            + a pd.DataFrame containing 2 columns 'source' and 'dest',
            the training data
        """

        corpus = self.get_corpus(data_path)

        tfidf = TFIDF(corpus,
                      vocab_save_path=vocab_save_path)

        with open(os.path.join(vocab_save_path, "word2index.json"), "r") as f:
            word2index = json.load(f)

        df = pd.Series(corpus)
        df = pd.DataFrame(df)
        df.columns = ["dest"]
        df.dest = df.dest.apply(lambda x: "|".join(x))

        df["source"] = df.dest.progress_apply(lambda x: self.get_important_word(x.split("|"), tfidf))
        
        df.dest = df.dest.apply(lambda x: ",".join(self.encode_line(x.split("|"), word2index)))
        df.source = df.source.apply(lambda x: ",".join(self.encode_line(x.split("|"), word2index)))

        count = df.dest.apply(lambda x: len(x.split(",")))

        df = df[count <= 50]

        return df