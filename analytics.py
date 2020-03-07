# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 11:10:50 2020

@author: mankadp
"""
import os
import pandas as pd
import readability
import numpy as np
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer


def counting_features(texts):
    pronounCount = np.zeros(len(texts)).reshape(-1,1)
    prepositionCount = np.zeros(len(texts)).reshape(-1,1)
    conjunctionCount = np.zeros(len(texts)).reshape(-1,1)

    complexWordsCount = np.zeros(len(texts)).reshape(-1,1)
    longWordsCount = np.zeros(len(texts)).reshape(-1,1)
    syllablesCount = np.zeros(len(texts)).reshape(-1,1)
#    type token ratio is the no of unique words divided by total words
    typeTokenRatio = np.zeros(len(texts)).reshape(-1,1)
    wordCount = np.zeros(len(texts)).reshape(-1,1)
    print(len(texts))

    for i,text in enumerate(texts):
#        print(text)
        score = readability.getmeasures(text,lang="en")
        sentenceInfo = score["sentence info"]
        wordUsage = score["word usage"]
        # word usages
        pronounCount[i] = wordUsage['pronoun']
        prepositionCount[i] = wordUsage['preposition']
        conjunctionCount[i] = wordUsage['conjunction']
        # sentence info
        complexWordsCount[i] = sentenceInfo['complex_words']
        longWordsCount[i] = sentenceInfo['long_words']
        syllablesCount[i] = sentenceInfo['syllables']
        typeTokenRatio[i] = sentenceInfo['type_token_ratio']
        wordCount[i] = sentenceInfo['words']

    # Combining all of them into one
    featureCounts = pd.DataFrame(data = np.concatenate((pronounCount,prepositionCount,conjunctionCount,complexWordsCount
                                           ,longWordsCount,syllablesCount,typeTokenRatio,wordCount),axis=1),
    columns=["pronounCount","prepositionCount","conjunctionCount","complexWordsCount","longWordsCount",
             "syllablesCount","typeTokenRatio","wordCount"])
    return featureCounts


def remove_stopWords(texts):


    for text in texts:
        pass


if __name__ == "__main__":
    data = pd.read_csv("train.csv")

    '''
    try:
        os.system("pip install -r requirements.txt")
        nltk.download('stopwords')
        os.system("python -m spacy download en_core_web_sm")
    except:
        pass
    '''

    feature = data.iloc[:,1]
    texts = data['text']

# =============================================================================
    # Consists of all the counted features
    featureCounts=counting_features(texts)
# =============================================================================


# =============================================================================
    # Stop words Removal
# =============================================================================

    nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
    #nlp_vec = spacy.load('en_vecs', parse = True, tag=True, #entity=True)
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')



# =============================================================================
#     Lemmatization of text
# =============================================================================













