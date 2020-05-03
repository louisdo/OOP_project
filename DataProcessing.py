import re


class DataProcessing():
    def __init__(self,
                 file_name_input,
                 file_name_output,
                 file_stopwords):
        self.file_stopwords = file_stopwords
        self.file_name_input = file_name_input
        self.file_name_output = file_name_output
        self.data = []
        self.stopwords = []

    def ReadData(self):
        file_open = open(self.file_name_input, mode='r', encoding='utf_8')
        for line in file_open:
            self.data.append(line.rstrip("\n"))
        file_open.close()
        return self.data

    def GetStopWords(self):
        file_stop_words = open("stopwords.txt", mode='r', encoding='utf-8')
        for word in file_stop_words:
            self.stopwords.append(word.rstrip("\n"))
        file_stop_words.close()
        return self.stopwords

    def WriteData(self):
        data_input = self.ReadData()
        stop_words = self.GetStopWords()
        file_ouput = open(self.file_name_output, mode='w', encoding='utf-8')
        for data in data_input:
            res = []
            words = re.split('\s', data)
            for word in words:
                for stop_word in stop_words:
                    if re.match(stop_word, word):
                        res.append(word)
            file_ouput.write(' '.join(res))
            file_ouput.write('\n')
        file_ouput.close()




STOCK = DataProcessing('Vnexpress.CLASSIFIED.VNINDEX.txt', 'output.txt', 'stopwords.txt')
STOCK.WriteData()
