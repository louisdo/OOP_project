import string
import pandas as pd
from pyvi import ViTokenizer
from collections import Counter
import csv




class DataGenerate():
    def __init__(self, file_input, file_output, file_stopword):
        self.file_input = file_input
        self.file_output = file_output
        self.file_stopword = file_stopword

    def read_file(self):
        f = open(self.file_input, "r", encoding="utf-8")
        data = f.read()
        f.close()
        return data

    def write_file(self):
        f = open(self.file_output, "w")
        f.write(self, "content")
        f.close()

    def split_doc(self):
        sentence =  self.read_file()
        sentList = sentence.split(".")
        return sentList


    """ clean data and remove stopword"""


    #list of sentences
    def clean_doc(self):
        result = []
        arrays = self.split_doc()
        for arr in arrays:
            doc = ViTokenizer.tokenize(arr)
            # lower
            doc = doc.lower()
            # split words
            tokens = doc.split()
            table = str.maketrans('', '', string.punctuation.replace("_", ""))
            tokens = [w.translate(table) for w in tokens]
            tokens = [word for word in tokens if word]

            # remove stopword
            text1 = pd.read_csv(self.file_stopword, sep=" ", encoding="utf-8")
            list_stopwords = text1['stopwords'].values
            pre_text = []
            for word in tokens:
                if word not in list_stopwords:
                    pre_text.append(word)
            result.append(pre_text)
        return result


    #Counter sentences
    def count_word(self):
        arr = Counter(self.clean_doc())
        return arr



    def file_source(self):
        arr = self.clean_doc()
        result = []
        for x in arr:
            tmp=""
            for i in x:
                tmp=tmp+i+"|"
            result.append(tmp)
        return result




    def write_filecsv_source(self):
        tmpArr1 = self.split_doc()
        tmpArr2 = self.file_source()
        fArr = []
        for i in range(len(tmpArr1)):
            tmparr = []
            tmparr.append(tmpArr1[i])
            tmparr.append(tmpArr2[i])
            fArr.append(tmparr)

        with open(self.file_output, "w", newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["dest", "source"])
            for j in fArr:
                writer.writerow([j[0], j[1]])




if __name__ == "__main__":
    p = DataGenerate('./datatest/input.txt',  './datatest/output.csv', './datatest/stopword.csv')
    print(p.split_doc())
    print(p.file_source())
    p.write_filecsv_source()