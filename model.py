from nltk import NaiveBayesClassifier as nbc
import pickle
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords
import codecs
from itertools import chain

a = thai_stopwords()
# pos.txt
with codecs.open('pos.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listpos=[e.strip() for e in lines]
del lines
f.close() # ปิดไฟล์
# neg.txt
with codecs.open('neg.txt', 'r', "utf-8") as f:
    lines = f.readlines()
listneg=[e.strip() for e in lines]
f.close() # ปิดไฟล์

pos1=['pos']*len(listpos)
neg1=['neg']*len(listneg)

training_data = list(zip(listpos,pos1)) + list(zip(listneg,neg1))

vocabulary = set(chain(*[(set(word_tokenize(i[0]))-set(thai_stopwords())) for i in training_data]))
#vocabulary = set(chain(*[x for x in a if x not in [list(set(word_tokenize(i[0]))) for i in training_data]]))

feature_set = [({i:(i in word_tokenize(sentence)) for i in vocabulary},tag) for sentence, tag in training_data]

classifier = nbc.train(feature_set)

with open('vocabulary.pkl', 'wb') as out_strm: 
    pickle.dump(vocabulary,out_strm)
out_strm.close()

with open('sentiment.pkl', 'wb') as out_strm: 
    pickle.dump(classifier,out_strm)
out_strm.close()

print('OK')