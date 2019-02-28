# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:32:15 2018

@author: Administrator
"""

# %%
import pandas as pd

path = './data'
# ted = pd.read_csv(path + '/netease_ted_comments.csv', encoding="utf-8")
domestic = pd.read_csv(path + '/netease_domestic_comments.csv', encoding="utf-8")
# foreign = pd.read_csv(path + '/netease_foreign_comments.csv', encoding="utf-8")
# %%
import pkuseg

def fenci(data, seg):
    #载入停用词
    with open('./data/stopwords.txt','r',encoding='utf-8') as f:
        stop = [line.strip() for line in f.readlines()]

    #遍历所有行进行分词
    tags = []
    for line in data:
        outstr = ''
        sen = seg.cut(line)
        for word in sen:
            if word not in stop:
                if word != '\t':
                    outstr += word
                    outstr += ' '
        tags.append(outstr)
    # jieba.analyse.set_stop_words(path + '/stopwords.txt')
    # tags = [' '.join(set(jieba.analyse.textrank(x,topK=100))) for x in data] #基于tf-idf算法提取关键词
    return tags

# %%
def judge1(l1, l2, s, s0, i):
    if l1 == l2:
        s0 = delete(l2, s0, i)  # 执行删除操作
        l2 = [s]
    else:
        l1, l2 = [s], []
    return l1, l2, s0


def judge2(l1, l2, s, s0, i):
    if l1 == l2 and len(l1) >= 2:
        s0 = delete(l2, s0, i)
        l1, l2 = [s], []
        # 执行删除操作
    elif l1 != l2 and not l2:
        l1.append(s)
    elif l1 != l2 and l2:
        l2.append(s)
    return l1, l2, s0


def delete(l2, s0, i):
    # 将删除项赋值为空格
    start = i - len(l2)
    s0[start:i] = ' ' * len(l2)
    return s0


def condense(s,forward=True):
    if forward:
        l1 = [s[0]]
        l2 = []
        s0 = list(s)
        for i in range(1, len(s)):
            next_ = s[i]
            if l1[0] == next_ and not l2:
                # 触发rule1
                l2.append(next_)
            elif l1[0] == next_ and l2:
                # 触发rule2,rule3
                l1, l2, s0 = judge1(l1, l2, next_, s0, i)
            elif l1[0] != next_:
                # 触发rule4,5,6
                l1, l2, s0 = judge2(l1, l2, next_, s0, i)
    else:
        l1 = [s[-1]]
        l2 = []
        s0 = list(s)
        for i in range(len(s)-1, 0, -1):
            next_ = s[i]
            if l1[0] == next_ and not l2:
                # 触发rule1
                l2.append(next_)
            elif l1[0] == next_ and l2:
                # 触发rule2,rule3
                l1, l2, s0 = judge1(l1, l2, next_, s0, i)
            elif l1[0] != next_:
                # 触发rule4,5,6
                l1, l2, s0 = judge2(l1, l2, next_, s0, i)

    s0 = [x.strip() for x in s0]
    return ''.join(s0)


def condense_list(l):
    '''机械压缩处理'''
    new_l = [condense(tags,forward=True) for tags in l] #前向
    new_l = [condense(tags,forward=False) for tags in new_l] #反向
    return new_l


def delete_short_tags(df, k=5):
    '''删除短评论，默认删除4个字以下'''
    l = df['Content']
    n = len(l)
    d = [i for i in range(n) if len(l[i]) >= k]
    return df.iloc[d,:]


if __name__ == '__main__':
    import datetime

    # n1 = ted.shape[0]
    n2 = domestic.shape[0]
    # n3 = foreign.shape[0]

    # print('Ted数据集原始样本数：%d'%n1)
    print('Domestic数据集原始样本数：%d'%n2)
    # print('Foreign数据集原始样本数：%d'%n3)

    t1 = datetime.datetime.now()
    # ted['Content'] = condense_list(ted['Content'])
    domestic['Content'] = condense_list(domestic['Content'])
    # foreign['Content'] = condense_list(foreign['Content'])

    # ted = delete_short_tags(ted)
    domestic = delete_short_tags(domestic)
    # foreign = delete_short_tags(foreign)
    print('机械压缩去词共用时：{}'.format(datetime.datetime.now()-t1))

    t1 = datetime.datetime.now()
    seg = pkuseg.pkuseg(user_dict=['TED','ted','Ted'])
    # ted['Content'] = fenci(ted['Content'], seg)
    domestic['Content'] = fenci(domestic['Content'], seg)
    # foreign['Content'] = fenci(foreign['Content'], seg)
    print('分词共用时：{} '.format(datetime.datetime.now()-t1))

    ###去重
    # ted = ted.drop_duplicates(keep=False).dropna(axis=0,how='any').reset_index(drop=True)
    domestic = domestic.drop_duplicates(keep=False).dropna(axis=0,how='any').reset_index(drop=True)
    # foreign = foreign.drop_duplicates(keep=False).dropna(axis=0,how='any').reset_index(drop=True)

    # ted = delete_short_tags(ted)
    domestic = delete_short_tags(domestic)
    # foreign = delete_short_tags(foreign)

    # print('Ted数据集处理后样本数：%d，无效评论率：%.2f' % (ted.shape[0],(n1-ted.shape[0])/ted.shape[0]))
    # print('Domestic数据集处理后样本数：%d，无效评论率：%.2f' % (domestic.shape[0], (n2 - domestic.shape[0]) / n2))
    # print('Foreign数据集处理后样本数：%d，无效评论率：%.2f' % (foreign.shape[0], (n3 - foreign.shape[0]) / n3))

    # ted.to_csv(path + '/pku_ted_tags.csv')
    domestic.to_csv(path + '/pku_domestic_tags_new.csv')
    # foreign.to_csv(path + '/pku_foreign_tags.csv')














