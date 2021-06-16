#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 17:13:52 2018

@author: onee

"""

# imports needed and set up logging
import gzip
import gensim 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data_file = "./cocoapi/coco2017/cocoword_all.txt.gz"#reviews_data.txt.gz"

#with gzip.open (data_file, 'rb') as f:
#    for i,line in enumerate (f):
#        print(line)
#        break

def read_input(input_file):
    """This method reads the input file which is in gzip format"""
    
    logging.info("reading file {0}...this may take a while".format(input_file))
    
    with gzip.open (input_file, 'rb') as f:
        for i, line in enumerate (f): 

            if (i%10000==0):
                logging.info ("read {0} reviews".format (i))
            # do some pre-processing and return a list of words for each review text
            yield gensim.utils.simple_preprocess (line)
            
documents = list (read_input (data_file))
logging.info ("Done reading data file")
    
            
# read the tokenized reviews into a list
# each review item becomes a serries of words
# so this becomes a list of lists
#documents = read_input (data_file)
#logging.info ("Done reading data file")

# Load Google's pre-trained Word2Vec model.
#model = gensim.models.Word2Vec.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)  

model = gensim.models.Word2Vec (documents, size=150, min_count=1, workers=10)
model.train(documents,total_examples=len(documents),epochs=1000)

w1 = "sky"
model.wv.most_similar (positive=w1, topn=10)

model.wv.similarity(w1="person",w2="dog")