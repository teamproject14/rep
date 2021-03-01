# -*- coding: utf-8 -*-
from gensim.models import word2vec
import pandas as pd

from src.mylib import logger


class w2v:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path

    def train_model(self):
        sentences = pd.read_csv(self.data_path)['Querys'].tolist()
        logger.info('正在训练w2v模型')
        model = word2vec.Word2Vec(sentences, sg=1)  # skip-gram
        logger.info(f'训练完毕，已保存: {self.save_path}')
        model.save(self.save_path)

    def load_model(self):
        logger.info(f'读取w2v 模型: {self.save_path}')
        model = word2vec.Word2Vec.load(self.save_path)
        return model
