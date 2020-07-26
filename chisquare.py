# import these modules
import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk import ngrams

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

