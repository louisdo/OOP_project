import re


def readData(file_name_input, file_name_output):
    """ Get data from file input
        Store sentences in each line in list lines """
    file_input = open(file_name_input, mode='r', encoding='utf-8')
    lines = []
    for line in file_input:
        lines.append(line.rstrip("\n"))
    file_input.close()

    """ Get the keywords from file stopwords
        Store keywords in list stop_words"""
    stop_words = []
    file_stop_word = open("stopwords.txt", mode='r', encoding='utf-8')
    for word in file_stop_word:
        stop_words.append(word.rstrip('\n'))
    file_stop_word.close()

    """Write the result to file output"""
    file_output = open(file_name_output, mode='w', encoding='utf-8')
    for line in lines:
        res = []
        words = re.split('\s', line)
        for word in words:
            for stop_word in stop_words:
                if re.match(stop_word, word):
                    res.append(word)
        file_output.write(' '.join(res))
        file_output.write('\n')
    file_output.close()


if __name__ == '__main__':
    readData('Vnexpress.CLASSIFIED.VNINDEX.txt', 'output.txt')
    #readData('Vnexpress.CLASSIFIED.ALL.txt', 'output1.txt')
