import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from model.classify_model import nnModel
import torch as tc

w2cModel = Word2Vec.load('./model/word2vec.model')
file = './data/pku_domestic_tags.csv'
df = pd.read_csv(file, encoding='utf-8').dropna()
tags = df['Content']
tags = [tag.split(' ')[:-1] for tag in tags.values]
###parameters###
batch_size = len(tags)
seq_len = 10
input_size = 100
hidden_size = 20
num_labels = 4
num_layers = 3
################
X = np.zeros((batch_size, seq_len, input_size))
for i in range(batch_size):
    vec = w2cModel.wv[tags[i]]
    if vec.shape[0]>seq_len:
        X[i, :, :] = vec[:seq_len, :]
    else:
        X[i, :vec.shape[0], :] = vec

y = tc.tensor(df['label'])
net = nnModel(input_size, seq_len, num_labels, num_layers)
##########Now Training###########
optimizer = tc.optim.Adam(net.parameters())
criterion = tc.nn.CrossEntropyLoss()
for step in range(10000):
    pred = net(X).squeeze()
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step%100==0:
        print('第{}次迭代，loss:{}'.format(step, loss))

tc.save(net, 'nnModel.pkl')

