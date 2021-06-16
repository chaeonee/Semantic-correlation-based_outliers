# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:40:33 2018

@author: onee
"""

import numpy as np
from numpy import median
from collections import Counter
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import gensim.models as g

g_classlist = []

model = g.KeyedVectors.load_word2vec_format('./cocoapi/coco2017/model/coco10_model.bin', binary=True)

def word_distance(w1,w2):
    wordname, wordindex = get_wordlist(g_classlist)
    
    word1 = wordname[int(w1)]
    word2 = wordname[int(w2)]
  
    if word1 in model.vocab:
        if word2 in model.vocab:
            dis = 1-model.wv.similarity(w1=word1,w2=word2)
            print(word1)
            print(word2)
        else:
            dis = 2
    else:
        dis = 2
    
    return round(dis,5)

def get_wordlist(classlist):
    classname = g_classlist
#    classindex = classlist[1]
    
    wordname = []
    wordindex = []
    
    for i in range(len(classname)):
        tmp = classname[i].replace('-',' ').split()
        for j in range(len(tmp)):
            if tmp[j] != 'other':
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
    
#    dist = []    
#    for i in range(len(wordname)):
#        temp = []
#        for j in range(len(wordname)):
#            temp.append(word_distance(i,j))
#        dist.append(list(temp))
#    
#    med = median(dist)
    
    clusters = DBSCAN(eps=0.3, min_samples=2, metric=word_distance).fit_predict(data.reshape(-1,1))
    plt.scatter(data,np.zeros_like(data),c=clusters, s=100)
    plt.show()
    
    data = []
    for i in range(len(wordname)):
        data.append([i,0])
        
    data = np.array(data, dtype=np.float64)
    
    df = pd.DataFrame(data, index=wordname, columns=['x', 'y'])
    df.shape
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    ax.scatter(df['x'],df['y'],c=clusters, s=100)
    
    for word, pos in df.iterrows():
        ax.annotate(word, pos, fontsize=9)    
    plt.show()
    
    cluster = list(set(clusters))
    outliers = []
    if len(cluster) > 1:
        if -1 in clusters:
#            print("-1있음")
            for i in range(len(clusters)):
                if -1 == clusters[i]:
                    outliers.append(i)
        else:
#            print("-1없음")
            for i in range(len(clusters)):
                cnt = Counter(clusters)
                min_cnt = min(cnt.values())
                
                if cnt[clusters[i]] == min_cnt:
                    outliers.append(i)
    
    outlierindex = []               
    for i in range(len(outliers)):
        outlierindex.append(wordindex[outliers[i]])
    
    result = []
    for i in range(len(outlierindex)):
        if outlierindex.count(outlierindex[i]) == wordindex.count(outlierindex[i]):
            result.append(outlierindex[i])
    result = list(set(result))
    
    for i in range(len(result)):
        result[i] = result[i] + 1
        
    return result

classlist = ['train', 'platform', 'railroad', 'sky-other', 'tree', 'metal', 'building-other', 'gravel', 'pavement', 'fence', 'textile', 'bridge']
#classlist = (['train', 'platform', 'railroad', 'sky-other', 'tree', 'metal', 'building-other', 'gravel', 'pavement', 'fence', 'traffic', 'bridge'], [[5, 12], [4, 4], [3, 3], [1, 2]])
result = get_outliers(classlist)
#print(result)            
