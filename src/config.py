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
idf_save_path = os.path.join(pretrain_path, 'idf_dict.dic')
"""日志地址"""
log_path = os.path.join(cur_dir, 'log')
"""特征数"""
age_num = 6
gender_num = 3
education_num = 6
"""训练参数"""
w2v_parameter = {
    'min_count': 2,
    'window': 5,
    'size': 300,
    'sample': 6e-05,
    'alpha': 0.03,
    'min_alpha': 0.0007,
    'negative': 15,
    'workers': 4,
    'iter': 30,
    'max_vocab_size': 50000,
    'sg': 1
}
fast_parameter = {
    'size': 300,
    'window': 3,
    'alpha': 0.03,
    'min_count': 2,
    'iter': 30,
    'max_n': 3,
    'word_ngrams': 2,
    'max_vocab_size': 50000
}
"""模型参数"""
nb_parameter = {
    'alpha': 0.001
}

lr_parameter = {
    'penalty': 'l2',
    'solver': 'sag',
    'multi_class': 'ovr'
}
