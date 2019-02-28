from functools import lru_cache
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
df = None


@lru_cache(maxsize=None)
def sorted_sibling_comments(x):
    sibling_comments = df.loc[df['CourseName'] == x]
    sibling_comments.sort_values('NaiveWeight', inplace=True)
    return sibling_comments


def naive_weight(row):
    return row['Up-votes'].astype('int64') - row['Down-votes'].astype('int64')


def weight(index, row):
    sorted_siblings = sorted_sibling_comments(row['CourseName'])
    indexes = np.where(sorted_siblings.index == index)
    siblings_count = len(sorted_siblings)
    if siblings_count == 1:
        return 1
    else:
        return (indexes[0] + 1) / len(sorted_siblings) / (len(sorted_siblings) - 1) * 2



def cal_weight(file):
    global df
    print('Calculating weight...')
    df = pd.read_csv(file)
    df['NaiveWeight'] = naive_weight(df)
    for index, row in df.iterrows():
        df.loc[index, 'Weight'] = weight(index, row)
    return df        



