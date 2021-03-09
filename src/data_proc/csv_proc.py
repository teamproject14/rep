# -*- coding: utf-8 -*-
import pandas as pd
from src.config import *
from src.mylib import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from collections import namedtuple
from tqdm import tqdm

tqdm.pandas()
age_feature = namedtuple('age_feature', ['age', 'feature'])
gender_feature = namedtuple('gender_feature', ['gender', 'feature'])
education_feature = namedtuple('education_feature', ['education', 'feature'])


def all_query_save():
    train_data = pd.read_csv(train_data_path, sep="###__###", header=None, encoding='utf-8')
    test_data = pd.read_csv(test_data_path, sep="###__###", header=None, encoding='utf-8')
    train_data.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']
    test_data.columns = ['ID', 'Query_List']
    train_data["query"] = train_data['Query_List'].progress_apply(query_clean)
    # train_data = train_data[["ID", "query"]]
    train_data.to_csv(os.path.join(data_path, 'train_query.csv'), index=False)
    test_data["query"] = test_data['Query_List'].progress_apply(query_clean)
    # test_data = test_data[["ID", "query"]]
    test_data.to_csv(os.path.join(data_path, 'test_query.csv'), index=False)

    query_data = pd.concat([train_data, test_data])
    query_data.to_csv(os.path.join(data_path, 'all_query.csv'), index=False)


def train_csv_process(path, show_log=False):
    data = pd.read_csv(path, sep="###__###", header=None, encoding='utf-8').dropna()
    data.columns = ['ID', 'Age', 'Gender', 'Education', 'Query_List']
    cond1 = data['Age'] != 0
    cond2 = data['Gender'] != 0
    cond3 = data['Education'] != 0
    data = data[cond1 & cond2 & cond3]
    save_path = path.split('.')
    save_path = save_path[0] + '_clean.' + save_path[1]
    data.to_csv(save_path, index=False)

    if show_log:
        logger.info("=" * 10 + "年龄分布" + "=" * 10)
        logger.info(data.Age.value_counts())
        logger.info("=" * 10 + "性别分布" + "=" * 10)
        logger.info(data.Gender.value_counts())
        logger.info("=" * 10 + "教育分布" + "=" * 10)
        logger.info(data.Education.value_counts())
        logger.info(data.shape)
    # data = data.head(5)

    data['Querys'] = data['Query_List'].progress_apply(query_clean)
    # 统计全局idf

    # d = data['Querys'].tolist()
    # cal_idf_dict(d, save_path=idf_save_path)

    data['Querys_keywords'] = data['Querys'].progress_apply(get_keywords)
    # print(data['Querys_keywords'].tolist()[0])

    # 特征保存
    save_column = ['Age', 'Gender', 'Education', 'Querys', 'Querys_keywords']
    save_data = data[save_column]
    save_data.to_csv(os.path.join(data_path, "train_data.csv"), index=False)

    # for sc in save_column:
    #     data_ = data[[sc, 'Querys', 'Querys_keywords']]
    #     data_.to_csv(os.path.join(data_path, f"{sc}.csv"), index=False)


if __name__ == '__main__':
    all_query_save()
    # train_csv_process(train_data_path, False)
    # train_csv_process(test_data_path, False)
    # all_query_save()
