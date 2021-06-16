#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 00:54:57 2018

@author: onee
"""
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

#model_name = '/Users/onee/cocoapi/coco2017/model/coco5000_model.bin'
#model = g.Doc2Vec.load(model_name)

model = g.KeyedVectors.load_word2vec_format('./cocoapi/coco2017/model/coco5000_model.bin', binary=True)  
model =  g.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
#f = open("/Users/onee/word2vec/cocolist.txt",'r')
#classlist = []
#t_class = []
#for line in f.readlines():
#    classes = line.strip().split()
#    for i in range(len(classes)):
#        t_class = classes[i].replace('-','\n')
#        t_class = t_class.split()
#       
#        for j in range(len(t_class)):
#            classlist.append(t_class[j])
#f.close()
#
#classlist = list(set(classlist))
#classlist.remove('other')
#classlist.remove('stuff')

f = open("./img2.txt",'r')
classlist = []
for line in f.readlines():
    classes = line.strip().split()
    
    for i in range(len(classes)):
        classlist.append(classes[i])
f.close()

#vocab = list(model.wv.vocab)
X = model[classlist]

#print(len(X))
#print(X[0][:10])
tsne = TSNE(n_components=2)

# 100개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X)
# X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=classlist, columns=['x', 'y'])
df.shape

fig = plt.figure()
fig.set_size_inches(10, 5)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

from scipy.spatial import distance

for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)
plt.show()

for i in range(0,8):
    for j in range(0,8):
        dst = distance.euclidean(X[i],X[j])
        print(dst)
    print('==================')

import sklearn
from sklearn.cluster import DBSCAN
from collections import Counter

colors = ['red','blue','yellow','green']

dbscan = DBSCAN(eps=0.1, min_samples=10).fit(X_tsne)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'],c=colors,s=12)

plt.show()