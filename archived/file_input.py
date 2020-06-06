import string

import pandas as pd

from pyvi import ViTokenizer
from collections import Counter

import csv



class DataProcessing():
    def __init__(self, input_path, output_path, file_stopword):
        self.input_path = input_path
        self.output_path = output_path
        self.file_stopword = file_stopword

    def read_file(self):
        in_file = open(self.input_path, "r", encoding="utf-8")
        content = in_file.read()
        in_file.close()
        return content

    def write_file(self):

        data = []
        arrays = self.split_doc()
        for arr in arrays:
            main_content = ViTokenizer.tokenize(arr)

            tokens = main_content.split()
            table = str.maketrans('', '', string.punctuation.replace("_", ""))
            tokens = [w.translate(table) for w in tokens]
            tokens = [word for word in tokens if word]

            text1 = pd.read_csv(self.file_stopword, sep=" ", encoding="utf-8")
            list_stopwords = text1['stopwords'].values
            pre_text = ""
            for word in tokens:
                if word not in list_stopwords:
                    pre_text += "," + word

            data.append((main_content + "|" + pre_text).split("|"))

        list_for_write = []

        fieldnames = "dest,source".split(",")

        for values in data:
            inner_dict = dict(zip(fieldnames, values))
            list_for_write.append(inner_dict)

        file_object = open(self.output_path, 'w', encoding="utf-8")

        csv_dict_file_writer = csv.DictWriter(file_object, fieldnames=fieldnames)

        csv_dict_file_writer.writeheader()

        for row in list_for_write:
            csv_dict_file_writer.writerow(row)

    def split_doc(self):
        sentence = self.read_file()
        sentList = sentence.split(".")
        return sentList


    def clean_doc(self):
        result = []
        arrays = self.split_doc()
        for arr in arrays:
            main_content = ViTokenizer.tokenize(arr)

            tokens = main_content.split()
            table = str.maketrans('', '', string.punctuation.replace("_", ""))
            tokens = [w.translate(table) for w in tokens]
            tokens = [word for word in tokens if word]

            text1 = pd.read_csv(self.file_stopword, sep=" ", encoding="utf-8")
            list_stopwords = text1['stopwords'].values
            pre_text = []
            for word in tokens:
                if word not in list_stopwords:
                    pre_text.append(word)
            result.append(pre_text)
        return result

    # Counter sentences
    def count_word(self):
        arr = Counter(self.clean_doc())
        return arr


#Test write to csv file
f_test = DataProcessing("input_data.txt", "output_data.csv", "stopword.csv")
f_test.write_file()