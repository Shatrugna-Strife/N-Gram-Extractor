import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


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


tokenizer = RegexpTokenizer(r'(?:[^\W\d_]\.)+| \d+(?:[.,]\d+)*(?:[.,]\d+)|\w+(?:\.(?!\.|$))?| \d+(?:[-\\/]\d+)*| \$')
# tokenizer = RegexpTokenizer(r'\w+')
'''
(?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    \d+(?:[.,]\d+)*(?:[.,]\d+)|       # numbers in format 999.999.999,99999
    \w+(?:\.(?!\.|$))?|               # words with numbers (including hours as 12h30),
                                      # followed by a single dot but not at the end of sentence
    \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
    \$|                               # currency sign
    -+|                               # any sequence of dashes
    \S                                # any non space characters
'''
f = open("wiki_06", 'r', encoding = "utf8").read()
data = re.sub(r'<.*?>', '', f)
tokenize = tokenizer.tokenize(data)
tokenize = [w.lower() for w in tokenize ]
fw = open("token_06_0.txt", 'w',encoding = "utf8")
fw.write(str(tokenize))
unigram = []
bigram = []
trigram = []

for i in range(1,4):
    f1 = open("token_06_0"+str(i)+".txt", 'w',encoding = "utf8")
    # gram = nltk.util.ngrams(tokenize,i)
    gram = nltk.ngrams(tokenize,i)
    temp = Counter(gram)
    f1.write(str(temp))
    if i == 1:
        print("Total Unique Unigrams:", len(temp))
        print("Total number of (most frequent) uni-grams are required to cover the 90% of the complete corpus:", no_of_words(90, list(temp.values())))
        for i, j in temp.items():
            unigram.append([str(i[0]),j])
    elif i == 2:
        print("Total Unique Bigrams:", len(temp))
        print("Total number of (most frequent) bi-grams are required to cover the 80% of the complete corpus:", no_of_words(80, list(temp.values())))
        for i, j in temp.items():
            bigram.append([str(i[0])+"-"+str(i[1]),j])
    else:
        print("Total Unique Trigrams:", len(temp))
        print("Total number of (most frequent) tri-grams are required to cover the 70% of the complete corpus:", no_of_words(70, list(temp.values())))
        for i, j in temp.items():
            trigram.append([str(i[0])+"-"+str(i[1])+"-"+str(i[2]),j])

ps = PorterStemmer()
data_stem = []

for w in tokenize:
    data_stem.append(ps.stem(w))

unigram_stem = []
bigram_stem = []
trigram_stem = []

fw = open("token_stem_06_0.txt", 'w',encoding = "utf8")
fw.write(str(data_stem))
for i in range(1,4):
    f1 = open("token_stem_06_0"+str(i)+".txt", 'w',encoding = "utf8")
    # gram = nltk.util.ngrams(tokenize,i)
    gram = nltk.ngrams(data_stem,i)
    temp = Counter(gram)
    f1.write(str(temp))
    if i == 1:
        print("Total Unique Unigrams(after stemming):", len(temp))
        print("Total number of (most frequent) uni-grams are required to cover the 90% of the complete corpus(after stemming):", no_of_words(90, list(temp.values())))
        for i, j in temp.items():
            unigram_stem.append([str(i[0]),j])
    elif i == 2:
        print("Total Unique Bigrams(after stemming):", len(temp))
        print("Total number of (most frequent) bi-grams are required to cover the 80% of the complete corpus(after stemming):", no_of_words(80, list(temp.values())))
        for i, j in temp.items():
            bigram_stem.append([str(i[0])+"-"+str(i[1]),j])
    else:
        print("Total Unique Trigrams(after stemming):", len(temp))
        print("Total number of (most frequent) tri-grams are required to cover the 70% of the complete corpus(after stemming):", no_of_words(70, list(temp.values())))
        for i, j in temp.items():
            trigram_stem.append([str(i[0])+"-"+str(i[1])+"-"+str(i[2]),j])


lemmatizer = WordNetLemmatizer()
data_lemma = []

for w in tokenize:
    data_lemma.append(lemmatizer.lemmatize(w))

unigram_lemma = []
bigram_lemma = []
trigram_lemma = []

fw = open("token_lemma_06_0.txt", 'w',encoding = "utf8")
fw.write(str(data_lemma))
for i in range(1,4):
    f1 = open("token_lemma_06_0"+str(i)+".txt", 'w',encoding = "utf8")
    # gram = nltk.util.ngrams(tokenize,i)
    gram = nltk.ngrams(data_lemma,i)
    temp = Counter(gram)
    f1.write(str(temp))
    if i == 1:
        print("Total Unique Unigrams(after lemmatization):", len(temp))
        print("Total number of (most frequent) uni-grams are required to cover the 90% of the complete corpus(after lemmatization):", no_of_words(90, list(temp.values())))
        for i, j in temp.items():
            unigram_lemma.append([str(i[0]),j])
    elif i == 2:
        print("Total Unique Bigrams(after lemmatization):", len(temp))
        print("Total number of (most frequent) bi-grams are required to cover the 80% of the complete corpus(after lemmatization):", no_of_words(80, list(temp.values())))
        for i, j in temp.items():
            bigram_lemma.append([str(i[0])+"-"+str(i[1]),j])
    else:
        print("Total Unique Trigrams(after lemmatization):", len(temp))
        print("Total number of (most frequent) tri-grams are required to cover the 70% of the complete corpus(after lemmatization):", no_of_words(70, list(temp.values())))
        for i, j in temp.items():
            trigram_lemma.append([str(i[0])+"-"+str(i[1])+"-"+str(i[2]),j])


tokenize = [w.lower() for w in tokenize ]
stop_words = set(stopwords.words('english')) 
filtered_sentence = [w for w in tokenize if not w in stop_words]

# bigram = ngrams(filtered_sentence, 2)

bigrams = nltk.collocations.BigramAssocMeasures()
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(filtered_sentence)

bigramChiTable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.chi_sq)), columns=['bigram','chi-sq']).sort_values(by='chi-sq', ascending=False)
bigramChiTable.to_csv ('Chi-square collocation.csv', index = None, header=True)
bigramFinder.apply_freq_filter(3)

fw = open("Top20bigrams.txt", 'w',encoding = "utf8")

fw.write("Using Student's t Test\n")
# print(bigramFinder.nbest(bigrams.student_t, 20))
fw.write(str(bigramFinder.nbest(bigrams.student_t, 20)))
fw.write('\n\n')

fw.write("Using Pointwise Mutual Exclusion(PMI) Test\n")
# print(bigramFinder.nbest(bigrams.pmi, 20))
fw.write(str(bigramFinder.nbest(bigrams.pmi, 20)))
fw.write('\n\n')

fw.write("Using Likelihood ratio Test\n")
# print(bigramFinder.nbest(bigrams.likelihood_ratio, 20))
fw.write(str(bigramFinder.nbest(bigrams.likelihood_ratio, 20)))
fw.write('\n\n')

fw.write("Using Chi-square Test\n")
# print(bigramFinder.nbest(bigrams.chi_sq, 20))
fw.write(str(bigramFinder.nbest(bigrams.chi_sq, 20)))
fw.write('\n\n')


unigram.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in unigram][:20],[x[1] for x in unigram][:20])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Unigram',fontsize=15)
plt.show()

bigram.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in bigram][:15],[x[1] for x in bigram][:15])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Bigram',fontsize=15)
plt.show()

trigram.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in trigram][:10],[x[1] for x in trigram][:10])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Trigram',fontsize=15)
plt.show()

unigram_stem.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in unigram_stem][:20],[x[1] for x in unigram_stem][:20])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Unigram(after stemming)',fontsize=15)
plt.show()

bigram_stem.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in bigram_stem][:15],[x[1] for x in bigram_stem][:15])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Bigram(after stemming)',fontsize=15)
plt.show()

trigram_stem.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in trigram_stem][:10],[x[1] for x in trigram_stem][:10])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Trigram(after stemming)',fontsize=15)
plt.show()

unigram_lemma.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in unigram_lemma][:20],[x[1] for x in unigram_lemma][:20])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Unigram(after lemmatization)',fontsize=15)
plt.show()

bigram_lemma.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in bigram_lemma][:15],[x[1] for x in bigram_lemma][:15])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Bigram(after lemmatization)',fontsize=15)
plt.show()

trigram_lemma.sort(key = lambda x:x[1], reverse = True)
plt.plot([x[0] for x in trigram_lemma][:10],[x[1] for x in trigram_lemma][:10])
plt.xlabel('Value',fontsize=15)
plt.ylabel('Frequency',fontsize=15)
plt.xticks(fontsize=6)
plt.yticks(fontsize=10)
plt.title('Trigram(after lemmatization)',fontsize=15)
plt.show()