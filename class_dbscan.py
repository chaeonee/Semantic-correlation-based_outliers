# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:40:33 2018

@author: onee
"""

import numpy as np
from numpy import median
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

import gensim.models as g


def word_distance(w1,w2):
    model = g.KeyedVectors.load_word2vec_format('./coco5000_model.bin', binary=True)
    f = open("C:/Users/SM-PC/Desktop/word2vec/img3.txt",'r')  
    classlist = []
    for line in f.readlines():
        classes = line.strip().split()
        for i in range(len(classes)):
            classlist.append(classes[i])
    f.close()
    
    word1 = classlist[int(w1)]
    word2 = classlist[int(w2)]
    dis = 1-model.wv.similarity(w1=word1,w2=word2)
    
    return round(dis,5)
    

model = g.KeyedVectors.load_word2vec_format('./coco5000_model.bin', binary=True)  
f = open("./img3.txt",'r')
    
classlist = []
for line in f.readlines():
    classes = line.strip().split()
    for i in range(len(classes)):
        classlist.append(classes[i])
    
f.close()
        
#data = np.array(classlist)

#X = model[classlist]
data = []
for i in range(len(classlist)):
    data.append(i)
    
dist = []
for i in range(len(classlist)):
    temp = []
    for j in range(len(classlist)):
        temp.append(word_distance(i,j))
    dist.append(list(temp))
           
med = median(dist)
    
data = np.array(data, dtype=np.float64)


clusters = DBSCAN(eps=med, min_samples=2, metric=word_distance).fit_predict(data.reshape(-1,1))
plt.scatter(data,np.zeros_like(data),c=clusters, s=100)
plt.show()

data = []
for i in range(len(classlist)):
    data.append([i,0])
    
data = np.array(data, dtype=np.float64)
    
df = pd.DataFrame(data, index=classlist, columns=['x', 'y'])
df.shape

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'],df['y'],c=clusters, s=100)

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=9)
plt.show()