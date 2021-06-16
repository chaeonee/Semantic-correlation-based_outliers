#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 14:28:55 2018

@author: onee

"""

from pycocotools.coco import COCO
import skimage.io as io
#import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='./cocoapi/coco2017'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['aa'])
imgIds = coco.getImgIds(catIds=catIds )

all_ann = ''

for i in range(4001,len(imgIds)):
    img = coco.loadImgs(imgIds[i])[0]
    I = io.imread(img['coco_url'])
    #plt.axis('off')
    #plt.imshow(I)
    #plt.show()
    
    annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps=COCO(annFile)
    
    # load and display caption annotations
    annIds = coco_caps.getAnnIds(imgIds=img['id'])
    anns = coco_caps.loadAnns(annIds)
    for j in range(len(anns)):
        all_ann += anns[j]['caption']+' '
    
    all_ann += '\n'
    #coco_caps.showAnns(anns)
    #plt.imshow(I)
    #plt.axis('off')
    #plt.show()
    
f = open("./cocoword4.txt", 'w')
f.write(all_ann)
f.close()


#merge txt data
f = open("./cocoapi/coco2017/cocoword_all.txt",'a')
for i in range(1,5):
    f1 = open("./cocoapi/coco2017/cocoword"+str(i)+".txt",'r')
    s = f1.read()
    f1.close()
    f.write(s)
f.close()