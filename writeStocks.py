import numpy as np
import re
import operator

file = open("Vnexpress_CLASSIFIED_VNINDEX.txt", "r", encoding="utf8" )
#file = open("VNINDEX_500-525.txt", "r", encoding="utf8" )
#file = open("test.txt", "r", encoding = "utf8" )
data = file.read()
file.close()

words = re.findall("\w+", data)
# print(words)
word_dict = {}
for wd in words:
    if wd not in word_dict:
        word_dict[wd] = 1
    else : word_dict[wd] += 1
# đếm số từ trong input
# print(word_dict)

def regularWord(numberWords, inputDict):
    returnList = []
    sorted_dict = dict(sorted(inputDict.items(), key=operator.itemgetter(1),reverse=True))   # sort dict decrease by 
    sorted_list = list(sorted_dict)
    for i in range(numberWords):
        returnList.append(sorted_list[i])
    return returnList

regular = regularWord(30,word_dict)
#print(regular)
addStr = ""
for i in regular:
    addStr = addStr + "|" + i 

text_process = "V.-Index| \d+/\d+| \d+.\d+| \S\d+.\d+\S\S| \d+|%" + addStr      
#những từ đặc biệt và những từ thông thường

print(text_process)
lines = re.split("\d+:", data)
processData = []
# processData sẽ là mảng 2 chiều
for line in lines:
    tmp_process_data = re.findall(text_process, line)
    processData.append(tmp_process_data)


#fileOut = open("out.txt","w", encoding="utf8")
#fileOut = open("out20-25.txt","w", encoding="utf8")
fileOut = open("output.txt", "w", encoding="utf8")
for infor in processData:
    for i in infor:
        text = str(i)
        fileOut.write(text + "  ")
    fileOut.write("\n")

# # for line in data:
# #     print(line + "----")


