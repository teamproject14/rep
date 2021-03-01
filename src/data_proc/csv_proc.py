# -*- coding: utf-8 -*-
import pandas as pd
from src.config import *
from src.mylib import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import namedtuple

age_feature = namedtuple('age_feature', ['age', 'feature'])
gender_feature = namedtuple('gender_feature', ['gender', 'feature'])
education_feature = namedtuple('education_feature', ['education', 'feature'])


def train_csv_process(path, show_log=False):
    data = pd.read_csv(path, sep="###__###", header=None, encoding='utf-8').dropna()
    data.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']
    cond1 = data['Age'] != 0
    cond2 = data['Gender'] != 0
    cond3 = data['Education'] != 0
    data = data[cond1 & cond2 & cond3]
    if show_log:
        logger.info("=" * 10 + "年龄分布" + "=" * 10)
        logger.info(data.Age.value_counts())
        logger.info("=" * 10 + "性别分布" + "=" * 10)
        logger.info(data.Gender.value_counts())
        logger.info("=" * 10 + "教育分布" + "=" * 10)
        logger.info(data.Education.value_counts())
        logger.info(data.shape)

    data['Querys'] = data['Query_List'].map(query_clean)
    # 统计全局idf
    # d = data['Querys'].tolist()
    # cal_idf_dict(d, save_path=os.path.join(pretrain_path, 'idf_dict.dic'))

    # 特征保存
    # save_column = ['Age', 'Gender', 'Education']
    # for sc in save_column:
    #     data_ = data[[sc, 'Querys']]
    #     data_.to_csv(os.path.join(data_path, f"{sc}.csv"), index=False)


if __name__ == '__main__':
    train_csv_process(train_data_path, False)
