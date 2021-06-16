# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:40:33 2018

@author: onee
"""

import numpy as np
from collections import Counter
import pandas as pd

from sklearn.cluster import DBSCAN

import gensim.models as g

import yaml
model_path = yaml.load(open('model_path.yml'))
model = g.KeyedVectors.load_word2vec_format(model_path['word2vec_model'], binary=True)

g_classlist = []

def word_distance(w1,w2):
    wordname, wordindex = get_wordlist(g_classlist)
    
    word1 = wordname[int(w1)]
    word2 = wordname[int(w2)]
  
    if word1 in model.vocab:
        if word2 in model.vocab:
            dis = 1-model.wv.similarity(w1=word1,w2=word2)
        else:
            dis = 2
    else:
        dis = 2
    
    return round(dis,5)

def get_wordlist(classlist):
    classname = g_classlist
    
    wordname = []
    wordindex = []
    
    for i in range(len(classname)):
        tmp = classname[i].replace('-',' ').split()
        for j in range(len(tmp)):
            if tmp[j] != 'other' and tmp[j] != 'thing' and tmp[j] != 'stuff':
                wordname.append(tmp[j])
                wordindex.append(i)
                
    return wordname, wordindex


def get_outliers(classlist):
    global g_classlist
    g_classlist = classlist
    wordname, wordindex = get_wordlist(g_classlist)

    data = []
    for i in range(len(wordname)):
        data.append(i)
    data = np.array(data, dtype=np.float64)
    
    
    clusters = DBSCAN(eps=0.25, min_samples=3, metric=word_distance).fit_predict(data.reshape(-1,1)) 
    cluster = list(set(clusters))
    
    outliers = []
    if len(cluster) > 1:
        if -1 in clusters:
            for i in range(len(clusters)):
                if -1 == clusters[i]:
                    outliers.append(i)
        else:
            cnt = Counter(clusters)
            min_cnt = min(cnt.values())
            
            for i in range(len(clusters)):
                if cnt[clusters[i]] == min_cnt:
                    outliers.append(i)
                    
                    
    outlierindex = []
    for i in range(len(outliers)):
        outlierindex.append(wordindex[outliers[i]])

    result = []
    for i in range(len(outlierindex)): #위에서 분리된 단어들은 갯수를 비교해 분리된 단어들이 모두 포함되어야 outlier
        if outlierindex.count(outlierindex[i]) == wordindex.count(outlierindex[i]):
            result.append(outlierindex[i])
    result = list(set(result))

    for i in range(len(result)):
        result[i] = result[i] + 1

    return result
