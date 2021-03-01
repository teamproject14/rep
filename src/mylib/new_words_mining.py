# -*- coding: utf-8 -*-
import jieba
from nlp_zero import *
from typing import *


def get_newwords_dict(data: List[str], save_path):
    f = Word_Finder(min_proba=1e-5)
    f.train(data)
    f.find(data)
    new_words = [w for w, _ in f.words.items() if len(w) > 1 and len(w) < 5 and len(jieba.lcut(w, HMM=False)) > 1]
    with open(save_path, 'w', encoding="utf-8") as f:
        for word in new_words:
            f.write(word + '\n')
