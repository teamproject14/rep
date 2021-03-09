# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from src.mylib import logger
from sklearn.metrics import accuracy_score, recall_score, f1_score


def get_score(train_label, test_label, train_predict_label,
              test_predict_label):
    train_acc = accuracy_score(train_label, train_predict_label)
    test_acc = accuracy_score(test_label, test_predict_label)
    recall = recall_score(test_label, test_predict_label, average='micro')
    f1 = f1_score(test_label, test_predict_label, average='weighted')
    return train_acc, test_acc, recall, f1


def grid_model_train(model, paramters, train_features, train_label):
    gsearch = GridSearchCV(model, param_grid=paramters, scoring='accuracy', cv=3, verbose=True)
    # gsearch = RandomizedSearchCV(model, param_distributions=paramters,scoring='accuracy',cv=3,verbose=True)
    gsearch.fit(train_features, train_label)
    logger.info(f"{model} best parameters: {gsearch.best_params_}")
    return gsearch


# def