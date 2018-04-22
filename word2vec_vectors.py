# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 21:25:59 2018

@author: vasant
"""
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim

lines = ""
# pos = list()
#word = list() 
# label = list()        
# Head = list()
def conllu(fp):
    global lines 
    #print("verify")
    returnList = []
    for line in fp.readlines():
        if(line[0] == "#"):
            continue
        wordList = line.split()
        returnList.append(wordList)

        if not wordList:
            temp_return = returnList
            returnList=[]
            for words in temp_return:
                if len(words) > 0:
                    lines = lines + " " + words[1] 
                else:
                    lines = lines + '\n' 
                
train_file = "en_ewt-ud-train.conllu"

with open(train_file,encoding="utf-8") as fp:
    conllu(fp)

sent = sent_tokenize(lines) 
for idx, s in enumerate(sent):
    sent[idx] = word_tokenize(s)
print("word vectors training ")
model = gensim.models.Word2Vec(sent, workers=4,size=50,min_count=1)
model.save("50d.word2vec_vectors")