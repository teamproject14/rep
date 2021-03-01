# -*- coding: utf-8 -*-
import os

cur_dir = os.path.split(os.path.abspath(__file__))[0]
work_dir = os.path.split(cur_dir)[0]

"""数据地址"""
data_path = os.path.join(work_dir, 'data')
train_data_path = os.path.join(data_path, 'train.csv')
test_data_path = os.path.join(data_path, 'test.csv')
pretrain_path = os.path.join(cur_dir, 'pretrain')
pretrain_words_dict = os.path.join(pretrain_path, 'new_word.dic')
"""日志地址"""
log_path = os.path.join(cur_dir, 'log')
"""特征数"""
age_num = 6
gender_num = 3
education_num = 6
