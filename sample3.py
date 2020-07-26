import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

ps = PorterStemmer()

tokenizer = RegexpTokenizer(r'\w+')

f = open("wiki_06", 'r', encoding = "utf8").read()
data = re.sub(r'<.*?>', '', f)
tokenize = tokenizer.tokenize(data)
data_lemma = []

for w in tokenize:
    data_lemma.append(lemmatizer.lemmatize(w))

fw = open("token_lemma_06_0.txt", 'w',encoding = "utf8")
fw.write(str(data_lemma))
for i in range(1,4):
    f1 = open("token_lemma_06_0"+str(i)+".txt", 'w',encoding = "utf8")
    # gram = nltk.util.ngrams(tokenize,i)
    gram = nltk.ngrams(data_lemma,i)
    f1.write(str(Counter(gram)))

xy = []
for i, j in Counter(data_lemma).items():
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
