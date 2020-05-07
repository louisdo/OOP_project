import numpy as np
import re
import operator
import math


class Process_raw_data():
    def __init__(self,
                 name_file_input,
                 name_file_output,
                 number_word_regular):
                 
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
            words = re.findall("\w+", inputString)
            word_dict = {}

            for wd in words:
                if wd not in word_dict:
                    word_dict[wd] = 1
                else : word_dict[wd] += 1

            return word_dict


    #spit and count 
    def first_process(self):
        self.data = self.read_file()

        paragraphs = self.data.split('NHÓM')
        
        list_word_dict = []

        for para in paragraphs:
            word_dict = self.count_word(para)

            list_word_dict.append(word_dict)
        
        return list_word_dict

    
    def sort_words(self, inputDict):
        sorted_dict = dict(sorted(inputDict.items(), key=operator.itemgetter(1),reverse=True))   # sort dict decrease by 

        return sorted_dict


    def tf_idf(self, t, max_t, D, d_t):
        return t / max_t * math.log10(D / (1 + d_t))


    # Main Processing
    def regular_words(self, list_word_dict):
        listRegular = []

        frequence_word_in_data = self.count_word(self.data)

        max_t = 0 
        D = len(list_word_dict)

        #count word t in paragraph 
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
        list_word_dict = self.first_process()
        regular = self.regular_words(list_word_dict)

        regular_string = ''

        for i in regular:
            regular_string = regular_string + '|' + i 

        text_process = 'V.-Index| \d+/\d+| \d+.\d+| \S\d+.\d+\S\S| \d+|%' + regular_string      
        #những từ đặc biệt và những từ thông thường

        lines = re.split("\d+:", self.data)

        processData = []
        # processData sẽ là mảng 2 chiều

        for line in lines:
            tmp_process_data = re.findall(text_process, line)
            processData.append(tmp_process_data)
        
        return processData


    def write_file(self, content):
        fileOut = open(self.name_file_output, "w", encoding="utf8")

        for infor in content:
            for i in infor:
                text = str(i)
                fileOut.write(text + "  ")

            fileOut.write("\n")


print("nhập số từ phổ biến")
number_word_regular = int(input())

VN_INDEX = Process_raw_data('Vnexpress_CLASSIFIED_VNINDEX.txt', 'output.txt',number_word_regular)

output_data = VN_INDEX.process_data()
VN_INDEX.write_file(output_data)

print("done")
    