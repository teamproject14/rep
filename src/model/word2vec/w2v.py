# -*- coding: utf-8 -*-
import os
import joblib
import gensim
from gensim import models
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.mylib import logger, STOP_WORDS
from src.config import w2v_parameter, fast_parameter, pretrain_path, data_path


class Embedding:

    def __init__(self):
        self.stopwords = STOP_WORDS
        self.key_column = 'query'

    def train(self):
        logger.info('train tfidf')
        count_vect = TfidfVectorizer(stop_words=self.stopwords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))
        self.tfidf = count_vect.fit(self.data[self.key_column])
        logger.info('train word2vec')
        self.data[self.key_column] = self.data[self.key_column].apply(lambda x: x.split(' '))
        self.w2v = models.Word2Vec(**w2v_parameter)
        self.w2v.build_vocab(self.data[self.key_column])
        self.w2v.train(self.data[self.key_column],
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)
        logger.info('train fasttext')
        self.fast = models.FastText(self.data[self.key_column], **fast_parameter)

        # logger.info('train lda')
        # id2word = gensim.corpora.Dictionary(self.data[self.key_column])
        # corpus = [id2word.doc2bow(text) for text in self.data[self.key_column]]
        # self.lda_model =

        # self.LDAmodel = LdaMulticore(corpus=corpus,
        #                              id2word=self.id2word,
        #                              num_topics=30,
        #                              workers=4,
        #                              chunksize=4000,
        #                              passes=7,
        #                              alpha='asymmetric')

    def saver(self):
        logger.info('save tfidf model')
        joblib.dump(self.tfidf, os.path.join(pretrain_path, 'emb/tfidf'))

        logger.info('save w2v model')
        self.w2v.wv.save_word2vec_format(os.path.join(pretrain_path, 'emb/w2v.bin'), binary=False)

        logger.info('save fast model')
        self.fast.wv.save_word2vec_format(os.path.join(pretrain_path, 'emb/fast.bin'), binary=False)

        # logger.info('save lda model')
        # self.LDAmodel.save(os.path.join(pretrain_path, 'emb/lda'))

    def load_data(self, data_path):
        logger.info("数据加载...")
        self.data = pd.concat([
            pd.read_csv(os.path.join(data_path, "train_query.csv")),
            pd.read_csv(os.path.join(data_path, "test_query.csv"))
        ])
        # self.data = pd.read_csv(os.path.join(data_path, "all_query.csv"))

        self.key_column = 'keywords' if 'keywords' in self.data.columns else self.key_column

    def load(self):
        logger.info('load tfidf model')
        self.tfidf = joblib.load(os.path.join(pretrain_path, 'emb/tfidf'))

        logger.info('load w2v model')
        self.w2v = models.KeyedVectors.load_word2vec_format(os.path.join(pretrain_path, 'emb/w2v.bin'),
                                                            binary=False)

        logger.info('load fast model')
        self.fast = models.KeyedVectors.load_word2vec_format(os.path.join(pretrain_path, 'emb/fast.bin'),
                                                             binary=False)

        # logger.info('load lda model')
        # self.lda = LdaModel.load(os.path.join(pretrain_path, 'emb/lda'))


if __name__ == '__main__':
    emb = Embedding()
    emb.load_data(data_path)
    emb.train()
    emb.saver()
