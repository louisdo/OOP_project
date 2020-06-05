import csv
import math
import re


class DataProcessing():
    def __init__(self,
                 file_name_input,
                 file_name_output,
                 number_word_regular):
        self.file_name_input = file_name_input
        self.file_name_output = file_name_output
        self.number_word_regular = number_word_regular
        self.data = ''

    def read_file(self):
        file = open(self.file_name_input, "r", encoding="utf8")
        self.data = file.read()
        file.close()

        return self.data

    def count_word(self, input_data):
        words = re.findall("\w+", input_data)
        word_dict = {}

        for word in words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1

        return word_dict

    def tf_idf(self, t, max_t, D, d_t):
        return t / max_t * math.log10(D / (1 + d_t))

    def sort_words(self, input_dict):
        sorted_dict = dict(
            sorted(input_dict.items(), key=lambda x: x[1], reverse=True))  # sort dict decrease by value

        return sorted_dict

    def regular_words(self, list_word_dict):

        list_word_regular = []
        frequence_word_in_data = self.count_word(self.data)

        max_t = 0
        D = len(list_word_dict)
        count_appear_d = {}

        for word in frequence_word_in_data:
            count_appear_d[word] = 0
            if frequence_word_in_data[word] > max_t:
                max_t = frequence_word_in_data[word]

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
            list_word_regular.append(regulars[i])

        return list_word_regular

    def first_process(self):

        self.data = self.read_file()
        paragraphs = self.data.split('NHÓM')

        list_word_dict = []

        for para in paragraphs:
            word_dict = self.count_word(para)
            list_word_dict.append(word_dict)

        return list_word_dict

    def process_data(self):
        list_word_dict = self.first_process()
        regular = self.regular_words(list_word_dict)
        print(regular)

        regular_string = ''

        for i in regular:
            regular_string = regular_string + '|' + i

        text_process = 'V.-Index|number|%' + regular_string
        lines = re.split("\d+:", self.data)

        processed_data = []

        for line in lines:
            characters_to_number = '\d+[,]\d+|\d+[.]\d+|\d+'
            line_no_numbers = re.sub(characters_to_number, 'number', line)
            tmp_processed_data = re.findall(text_process, line_no_numbers)
            processed_data.append(tmp_processed_data)

        return processed_data

    def write_fileCSV(self, content):
        file_out = open(self.file_name_output, "w", encoding="utf8")

        list_line_source = []

        for line in content:
            tmp = ''
            for word in line:
                tmp = tmp + word + '|'

            list_line_source.append(tmp)

        list_line_dest = []

        lines = re.split('\d+:', self.data)

        for line in lines:
            line_dest = re.sub('\s+', '|', line)
            list_line_dest.append(line_dest)

        list_write = [['dest', 'source']]

        for i in range(len(list_line_source)):
            tmp = []
            tmp.append(list_line_dest[i])
            tmp.append(list_line_source[i])

            list_write.append(tmp)

        with file_out:
            writer = csv.writer(file_out)
            writer.writerows(list_write)

        file_out.close()


if __name__ == '__main__':
    print("Enter number of the most regular words: ")
    number_word_regular = int(input())
    stock = DataProcessing('Vnexpress.CLASSIFIED.VNINDEX.txt', 'output.csv', number_word_regular)

    output_data = stock.process_data()
    stock.write_fileCSV(output_data)
