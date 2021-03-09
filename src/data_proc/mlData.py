# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.mylib import logger
from src.model import Embedding
from src.config import *
from collections import namedtuple
import pickle
import joblib

train_data = namedtuple('train_data', ['ID', 'Age', 'Gender', 'Education', 'tfidf', 'w2v', 'fast'])
test_data = namedtuple('test_data', ['ID', 'tfidf', 'w2v', 'fast'])


class MLData:
    feature_method = ["w2v", "fast", "tfidf"]
    label_feature = ['Age', 'Gender', 'Education']
    w2v_method = ['mean', 'max']

    def __init__(self):
        self.em = Embedding()
        self.em.load()
        self.data_list = []

    def preprocessor(self):
        logger.info('加载数据')
        self.train = pd.read_csv(os.path.join(data_path, 'train_query.csv'))
        logger.info('计算训练集数据特征...')

        self.process_data(self.train, True)
        self.test = pd.read_csv(os.path.join(data_path, 'test_query.csv')).dropna()
        logger.info('计算测试集数据特征...')
        self.process_data(self.test, False)

    def process_data(self, data, is_train_data):
        # print(data.columns)  # ['ID', 'Age', 'Gender', 'Education', 'Query_List', 'query']

        times = 0
        save = 'train' if is_train_data else 'test'
        for idx, data in data.iterrows():
            d = {'ID': data.ID}
            for method in self.feature_method[:]:
                self.get_feature(data, method, d)
            if is_train_data:
                for lf in self.label_feature:
                    d[lf] = getattr(data, lf)

                self.data_list.append(train_data(**d))
            else:
                self.data_list.append(test_data(**d))
            if (idx + 1) % 50000 == 0:
                save_path = os.path.join(data_path, f"{save}_feature_{times}")
                logger.info(f"已存{idx+1}条{save}数据")
                self.save(save_path)
                self.data_list = []
                times += 1
        save_path = os.path.join(data_path, f"{save}_feature_{times}")
        self.save(save_path)
        self.data_list.clear()

    def get_feature(self, data, method, d: dict):
        sentence = data.query
        if method == 'tfidf':
            # data_ = [q for q in data["query"]]
            d[method] = self.em.tfidf.transform([sentence])
        elif method == 'w2v':
            d[method] = self.wam(sentence, self.em.w2v, agg=False)

        elif method == "fast":
            d[method] = self.wam(sentence, self.em.fast, agg=False)

    @staticmethod
    def wam(sentence, w2v_model, method='mean', agg=False):
        '''
        通过word average/max model 生成句向量
        '''
        arr = np.array([
            w2v_model.wv.get_vector(s) for s in sentence.split(" ")
            if s in w2v_model.wv.vocab.keys()
        ])
        if not agg:
            return arr
        if len(arr) > 0:
            if method == 'mean':
                return np.mean(np.array(arr), axis=0)
            else:
                return np.max(np.array(arr), axis=0)

        else:
            return np.zeros(300)

    def save(self, path):
        logger.info("特征存储")
        # pickle.dump(self.data_list, open(path, 'wb')) # memoryerror
        joblib.dump(self.data_list, open(path, 'wb'))


def statistics_feature():
    ml = MLData()
    ml.preprocessor()


if __name__ == '__main__':
    # statistics_feature()
    data = joblib.load(open(os.path.join(data_path, 'train_feature_0'), 'rb'))
    print(data[0])
