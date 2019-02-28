# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:53:18 2018

@author: Administrator
"""
import pandas as pd
from gensim.models import Word2Vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file = './data/pku_domestic_tags.csv'
df = pd.read_csv(file, encoding='utf-8').dropna()
tags = df['Content']
tags = [tag.split(' ')[:-1] for tag in tags.values]

model = Word2Vec(tags, sg = 1, size = 100, alpha =  0.05, window = 5, min_count = 1)
model.train(tags, total_examples=len(tags), epochs=1000)
print('Training complete!')
model.save('./model/word2vec.model')
print('Successfully saved!')



