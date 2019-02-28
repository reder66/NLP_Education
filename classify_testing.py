import torch as tc
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

X = np.load('./data/X.npy')
file = './data/pku_domestic_tags.csv'
df = pd.read_csv(file, encoding='utf-8').dropna()
y = df['label'].values
net = tc.load('nnModel.pkl')
pred = net(X).squeeze()
y_pred = tc.max(pred, 1)[1].numpy()
print(classification_report(y_pred, y))

# ######hist plot############
# plt.title('Predicted Label Hist Plot')
# sns.distplot(y_pred, hist=True, rug=True)
# plt.savefig('predict.png')
# plt.show()
#
# plt.title('True Label Hist Plot')
# sns.distplot(y, hist=True, rug=True)
# plt.savefig('true.png')
# plt.show()
# ######CM Plot##############
# cm = confusion_matrix(y_pred, y)
# print('Confusion Matrix:\n')
# print(cm)
# sns.heatmap(cm)
# plt.savefig('cm.png')
# plt.show()
