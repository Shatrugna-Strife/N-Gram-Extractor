import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re

def no_of_words(percent, list):
    temp = list[:]
    temp.sort(reverse = True)
    total = sum(temp)
    tmp = (percent*total)/100
    Sum =  0
    words = 0
    for i in temp:
        if Sum>tmp:
            break
        Sum += i
        words += 1
    return words



tokenizer = RegexpTokenizer(r'\w+')

f = open("wiki_06", 'r', encoding = "utf8").read()
data = re.sub(r'<.*?>', '', f)
data = tokenizer.tokenize(data)
tokenize = Counter(data)
print(len(tokenize))
Y = list(tokenize.values())
xy = []
for i, j in tokenize.items():
    xy.append([i,j])

xy.sort(key = lambda x:x[1], reverse = True)
Y.sort(reverse = True)
print(no_of_words(90, Y))
plt.plot([x[0] for x in xy][:20],[x[1] for x in xy][:20])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.ylabel('Frequency',fontsize=15)
plt.title('Fuck Off',fontsize=15)
plt.show()
