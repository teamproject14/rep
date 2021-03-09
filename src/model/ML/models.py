# -*- coding: utf-8 -*-
import os
from src.mylib import logger
from src.config import *
import joblib
import lightgbm as lgb
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from glob import glob
from src.data_proc import train_data, test_data, MLData


class Models:

    def __init__(self):
        self.model_dict = {}
        # self.model_add()
        self.train_data, self.train_label = self.load_data()

    def train(self):
        ...

    def save(self):
        ...

    def unbalance_helper(self):
        ...

    def data_process(self):
        ...

    def load_model(self):
        for p in os.listdir():  # todo
            model_name = None
            model = self.model_dict.get(model_name, None)

    def model_add(self):
        logger.info("添加模型...")
        nb_model = MultinomialNB(
            **nb_parameter
        )

        lr_model = LogisticRegression(**lr_parameter)
        self.model_dict['nb'] = nb_model
        self.model_dict['lr'] = lr_model

    def load_data(self):
        logger.info("加载数据")
        train_data_list = []
        train_label_list = []
        for path in glob(data_path + '/train_feature_*'):
            datas = joblib.load(open(path, 'rb'))
            for data in datas:
                feature_d = {}
                label_d = {}
                for fm in MLData.feature_method:
                    feature_d[fm] = getattr(data, fm)
                for lm in MLData.label_feature:
                    label_d[lm] = getattr(data, lm)
                print(label_d)
                train_data_list.append(feature_d)
                train_label_list.append(label_d)
        return train_data_list, train_label_list


if __name__ == '__main__':
    m = Models()
