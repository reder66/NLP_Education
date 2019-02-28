###############################构建指标######################################
import numpy as np
import pandas as pd

'''
columns:'CourseName', 'CourseURL', 'Content', 'Down-votes',
       'Up-votes', 'User'
'''
# pd.set_option('display.max_columns',300)
# def cal_cri(data):
#     '''同一课程下，importance = exp((1+up_i+down_i)/max_votes)'''
#     group = data.groupby('CourseName')[['Up-votes','Down-votes']]
#     all_votes = group.sum()
#     all_votes['max-votes'] = all_votes[['Up-votes','Down-votes']].apply(lambda x:x.max(),axis=1)
#     all_votes = all_votes.reset_index()[['CourseName','max-votes']]
#     data = pd.merge(data,all_votes,left_on = 'CourseName',right_on='CourseName')[['CourseName','Content','Up-votes','Down-votes','max-votes']]
#     data['importance'] = np.exp(data[['Up-votes','Down-votes']].apply(lambda x:x.sum()+1,axis=1)/data['max-votes'])
#     return data
#
#
# file_name = 'ted_tags'
# path = '.\data'
# file = pd.read_csv(path+'\\%s.csv'%file_name,encoding='utf-8')
# print(cal_cri(file))
##############################构建指标######################################