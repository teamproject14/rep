# -*- coding: utf-8 -*-
from typing import *
import jieba.posseg
import jieba.analyse
import math
from .word_tools import *
from ..config import pretrain_words_dict

jieba.load_userdict(pretrain_words_dict)


def text_cut(text):
    if phone_compile.search(text):
        # return ['手机号']
        return []
    if email_compile.search(text) or url_compile.search(text):
        # return ['网页']
        return []
    res = []
    words = jieba.posseg.lcut(text)
    for word, tag in words:
        if any(tag.find(i) >= 0 for i in allowPOS):
            res.append(word)

    return res


def query_clean(ql):
    res = []
    texts = ql.split("\t")
    for text in texts:
        res.extend(text_cut(text))
    res = [i for i in res if i not in STOP_WORDS and len(i) > 1]
    return "\t".join(res)


def get_keywords(sentence, topk=20, idf_path=None):
    tfidf = jieba.analyse.TFIDF(idf_path=idf_path)
    return tfidf.extract_tags(sentence, topK=topk, allowPOS=allowPOS)


def cal_idf_dict(words: List[List], save_path=None):
    """计算IDF"""
    idf_freq = {}
    for word in words:
        word = word.split("\t")
        cw = Counter(word)
        for w in cw:
            idf_freq[w] = idf_freq.get(w, 0.0) + 1.0
    length = len(words)
    for k, v in idf_freq.items():
        idf_freq[k] *= math.log(length / (v + 1)) / v

    if save_path:
        with open(save_path, 'w', encoding="utf-8") as f:
            for k, v in idf_freq.items():
                f.write(f"{k}\t{v}\n")
    return idf_freq
