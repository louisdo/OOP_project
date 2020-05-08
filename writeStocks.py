import numpy as np
import re
import operator
import math


class ProcessRawData():
    """
    Change normal sentences to keyword phrase
    """
    def __init__(self,
                 name_file_input: str,
                 name_file_output: str,
                 number_word_regular: int):
        """
        number_word_regular : limit the number of common words
        """
                 
        self.name_file_input = name_file_input
        self.name_file_output = name_file_output
        self.number_word_regular = number_word_regular
        self.data = ''
    

    def read_file(self):
        file = open(self.name_file_input, "r", encoding="utf8" )
        self.data = file.read()
        file.close()
        
        return self.data


    #count number of words in a paragraph
    def count_word(self, inputString):
        """
        inputString: str  (pragraph,sentences,...)
        Output: dictionary (key: word, value: number word in inputString)
        """

        words = re.findall("\w+", inputString)
        word_dict = {}

        for wd in words:
            if wd not in word_dict:
                word_dict[wd] = 1
            else : word_dict[wd] += 1

        return word_dict


    def first_process(self):
        """
        Split data by word (NHÓM) and count_word for each element 
        Output: list of dictionary count_word
        """
        self.data = self.read_file()

        paragraphs = self.data.split('NHÓM')
        
        list_word_dict = []

        for para in paragraphs:
            word_dict = self.count_word(para)

            list_word_dict.append(word_dict)
        
        return list_word_dict

    
    def sort_words(self, inputDict):
        sorted_dict = dict(sorted(inputDict.items(), key=operator.itemgetter(1),reverse=True))   # sort dict decrease by value

        return sorted_dict


    def tf_idf(self, t, max_t, D, d_t):
        return t / max_t * math.log10(D / (1 + d_t))


    # Main Processing
    def regular_words(self, list_word_dict):
        """
        list_word_dict: list of dictionary count_word
        Output: list of common words (nunmber_word_regular elements)
        """
        listRegular = []

        frequence_word_in_data = self.count_word(self.data)

        max_t = 0 
        D = len(list_word_dict)

        #count paragraph contain word t
        count_appear_d = {} 
        
        for word in frequence_word_in_data:
            count_appear_d[word] = 0

            if frequence_word_in_data[word] > max_t:
                max_t = frequence_word_in_data[word]

        #calculate count_appear_d
        for word_dict in list_word_dict:
            for w in word_dict:
                count_appear_d[w] += 1
        
        regular_dict = {}

        # calculate IF-IDF of words
        for word in frequence_word_in_data:
            t = frequence_word_in_data[word]
            d_t = count_appear_d[word] 

            regular_dict[word] = self.tf_idf(t, max_t, D, d_t)

        regular_dict = self.sort_words(regular_dict)
        regulars = list(regular_dict)

        for i in range(self.number_word_regular):
            listRegular.append(regulars[i])

        return listRegular


    def process_data(self):
        """
        Use regular expression to find common words
        Output: list of list common words
        """
        list_word_dict = self.first_process()
        regular = self.regular_words(list_word_dict)
        print(regular)

        regular_string = ''

        for i in regular:
            regular_string = regular_string + '|' + i 

        # combine with some special words
        text_process = 'V.-Index| \d+/\d+| \d+.\d+| \S\d+.\d+\S\S| \d+|%' + regular_string      

        lines = re.split("\d+:", self.data)

        processedData = []
        # processData sẽ là mảng 2 chiều

        for line in lines:
            tmp_processed_data = re.findall(text_process, line)

            processedData.append(tmp_processed_data)
        
        return processedData


    def write_file(self, content):
        """
        content: list of list
        """
        fileOut = open(self.name_file_output, "w", encoding="utf8")

        for infor in content:
            for i in infor:
                text = str(i)
                fileOut.write(text + "  ")

            fileOut.write("\n")


print("nhập số từ phổ biến")
number_word_regular = int(input())

VN_INDEX = ProcessRawData('Vnexpress_CLASSIFIED_VNINDEX.txt', 'output.txt',number_word_regular)

output_data = VN_INDEX.process_data()
VN_INDEX.write_file(output_data)

print("done")
    