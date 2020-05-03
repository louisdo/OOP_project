import numpy as np
import re
import operator


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

    
    #count words
    def first_process(self):
        self.data = self.read_file()
        
        words = re.findall("\w+", self.data)
        word_dict = {}

        for wd in words:
            if wd not in word_dict:
                word_dict[wd] = 1
            else : word_dict[wd] += 1

        return word_dict

    
    def regularWords(self, inputDict):
        returnList = []

        sorted_dict = dict(sorted(inputDict.items(), key=operator.itemgetter(1),reverse=True))   # sort dict decrease by 
        sorted_list = list(sorted_dict)

        for i in range(self.number_word_regular):
            returnList.append(sorted_list[i])

        return returnList


    def process_data(self):
        word_dict = self.first_process()
        regular = self.regularWords(word_dict)
        
        regular_string = ""

        for i in regular:
            regular_string = regular_string + "|" + i 

        text_process = "V.-Index| \d+/\d+| \d+.\d+| \S\d+.\d+\S\S| \d+|%" + regular_string      
        #những từ đặc biệt và những từ thông thường

        lines = re.split("\d+:", self.data)
        processedData = []
        # processData sẽ là mảng 2 chiều

        for line in lines:
            tmp_processed_data = re.findall(text_processed, line)
            processedData.append(tmp_processed_data)
        
        return processedData


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
    
    
