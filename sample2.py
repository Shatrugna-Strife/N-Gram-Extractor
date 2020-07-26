import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

ps = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')

f = open("wiki_06", 'r', encoding = "utf8").read()
data = re.sub(r'<.*?>', '', f)
tokenize = tokenizer.tokenize(data)
data_stem = []

for w in tokenize:
    data_stem.append(ps.stem(w))

fw = open("token_stem_06_0.txt", 'w',encoding = "utf8")
fw.write(str(data_stem))
for i in range(1,4):
    f1 = open("token_stem_06_0"+str(i)+".txt", 'w',encoding = "utf8")
    # gram = nltk.util.ngrams(tokenize,i)
    gram = nltk.ngrams(data_stem,i)
    f1.write(str(Counter(gram)))

xy = []
for i, j in Counter(data_stem).items():
    xy.append([i,j])

xy.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in xy][:20],[x[1] for x in xy][:20])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=8)
plt.yticks(fontsize=10)
plt.ylabel('Frequency',fontsize=15)
plt.title('Fuck Off',fontsize=15)
plt.show()
