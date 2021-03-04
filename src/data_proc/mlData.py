# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from src.mylib import logger
from src.model import Embedding
from src.config import *


class MLData:
    w2v_method = ['mean', 'max']

    def __init__(self):
        self.em = Embedding()
        self.em.load()

    def preprocessor(self):
        logger.info('加载数据')
        self.train = pd.read_csv(os.path.join(data_path, 'try_query.csv')).head(3)
        logger.info('计算训练集数据特征...')
        self.process_data(self.train)
        self.test = pd.read_csv(os.path.join(data_path, 'test_query.csv')).head(3)
        logger.info('计算测试集数据特征...')
        self.process_data(self.test)

    def process_data(self, data):
        for method in ["word2vec", "fasttext", "tfidf"]:
            self.get_feature(data, method)

    def get_feature(self, data, method):
        if method == 'tfidf':
            data_ = [q for q in data["query"]]

            data[method] = self.em.tfidf.transform(data_)
        elif method == 'word2vec':
            for m in self.w2v_method:
                data[f"{method}_{m}"] = np.vstack(data['query'].apply(
                    lambda x: self.wam(x, self.em.w2v, method=m)))
        else:
            for m in self.w2v_method:
                data[f"{method}_{m}"] = np.vstack(data['query'].apply(
                    lambda x: self.wam(x, self.em.fast)))

    @staticmethod
    def wam(sentence, w2v_model, method='mean'):
        '''
        通过word average/max model 生成句向量
        '''
        arr = np.array([
            w2v_model.wv.get_vector(s) for s in sentence.split(" ")
            if s in w2v_model.wv.vocab.keys()
        ])
        print(123, arr.shape)
        if len(arr) > 0:
            if method == 'mean':
                return np.mean(arr, axis=0)
            else:
                return np.max(arr, axis=0)

        else:
            return np.zeros(300)

    def save(self):
        logger.info("特征存储")
        self.train.to_csv('123.csv', index=False)
        self.test.to_csv('1234.csv', index=False)


def statistics_feature():
    ml = MLData()
    ml.preprocessor()
    ml.save()


if __name__ == '__main__':
    statistics_feature()
    # 报错# ValueError: Wrong number of items passed 300, placement implies 1

