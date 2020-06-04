import re
import logging
import json
import os
import pandas as pd
from pyvi import ViTokenizer
from tqdm import tqdm
from lib import TFIDF, Utils

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
    def clean_string(string: str):
        string = string.strip(".").replace(", ", " , ")
        char2replace = ["(", ")", "[", "]",
                        "{", "}", '""', "''",
                        "``", ]
        for character in char2replace:
            string = string.replace(character, "")
        string = string.lower()
        string = string.replace("vn-index", "vnindex")

        return string

    @staticmethod
    def process_sentence(sentence: str, patterns):
        """
        process a raw sentence

        input:
            + sentence: a raw sentence
            + patterns: a dictionary with its keys are the type name
            of the pattern and the correspoding values are the regex pattern
            of that type
        
        output:
            + processed_sentence, a string
            + a dictionary containing replaced terms

        Example:
            test = "Kết phiên sáng với 160 mã nhuộm xanh, Vn-Index tăng 9,76 điểm, lên 539,74 điểm, \
            mua bán gần 60 triệu cổ phiếu, ứng với hơn 976 tỷ đồng 1.2"
            patterns = RawDataPrep.get_patterns()

            print(RawDataPrep.process_sentence(test, patterns))

            output:
            ('kết phiên sáng với number_1 mã nhuộm xanh , vnindex tăng number_2 điểm , lên number_3 điểm , mua bán gần word_number_1 cổ phiếu , ứng với hơn word_number_2 đồng number_4',
            {'word_number_1': '60 triệu',
            'word_number_2': '976 tỷ',
            'number_1': '160',
            'number_2': '9,76',
            'number_3': '539,74',
            'number_4': '1.2'})
        """

        string = RawDataPrep.clean_string(sentence)

        pattern_count = {pattern: 0 for pattern in patterns}

        replaced_terms = {}

        pattern_types = ["date", "percent", "word_number", "time", "number"]

        for pattern in pattern_types:
            terms_found = [item[0] for item in list(patterns[pattern].finditer(string))]
            for index in range(len(terms_found)):
                replaced_terms[pattern + "_{}".format(index + 1)] = terms_found[index]
            string = patterns[pattern].sub(pattern, string)

        string = string.split(" ")

        for index in range(len(string)):
            term = string[index]
            if term in pattern_types:
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
    def get_data_from_file(data_path: str) -> list:
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

        patterns = RawDataPrep.get_patterns()

        processed_data = []

        for index in tqdm(range(len(data)), "Getting data"):
            string = data[index].split(":")[1].strip()
            processed_string = RawDataPrep.process_sentence(string, patterns)[0]
            if processed_string is not None:
                processed_data.append(processed_string)

        return processed_data

        
    @staticmethod
    def get_corpus(data_path: str):
        """
        input:
            + data_path: path to txt file containing raw data
        output:
            + a list of list which is the corpus
        """
        corpus = []
        
        data = RawDataPrep.get_data_from_file(data_path)

        for sentence in tqdm(data, desc="Tokenizing data"):
            tokenized_sentence = ViTokenizer.tokenize(sentence)
            tokens = tokenized_sentence.split(" ")
            
            corpus.append(tokens)

        return corpus

    @staticmethod
    def get_training_data(data_path: str, 
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
        # get corpus from raw data
        corpus = RawDataPrep.get_corpus(data_path)

        tfidf = TFIDF(corpus,
                      vocab_save_path=vocab_save_path)

        word2index = Utils.load_vocab(vocab_folder = vocab_save_path)[1]

        df = pd.Series(corpus)
        df = pd.DataFrame(df)
        df.columns = ["dest"]
        df.dest = df.dest.apply(lambda x: "|".join(x))

        # add a source column containing the most important terms from one sentence according to TF-IDF score
        df["source"] = df.dest.progress_apply(lambda x: RawDataPrep.get_important_word(x.split("|"), tfidf))
        
        token_encoder = lambda x: ",".join(Utils.encode_line(x.split("|"), word2index))
        df.dest = df.dest.apply(token_encoder)
        df.source = df.source.apply(token_encoder)

        count = df.dest.apply(lambda x: len(x.split(",")))

        # only take sentences with number of terms smaller than 50
        df = df[count <= 50]

        return df