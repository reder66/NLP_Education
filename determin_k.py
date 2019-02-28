import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

data = pd.read_csv('./data/pku_domestic_tags.csv')
tags = data['Content']
tags = [len(word.split(' ')[:-1]) for word in tags]
counter = Counter(tags)
sorted_counter = sorted(counter.items(), key=lambda x:x[1], reverse=True)
print(sorted_counter[:5])

plt.title('Word Number of Every Tgas')
plt.xlabel('word num')
plt.ylabel('frequency')
sns.distplot(tags, bins=50, hist=True, rug=True)
plt.savefig('determin_k.png')
plt.show()

