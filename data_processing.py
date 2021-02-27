import pandas as pd
import jieba
import re
train = pd.read_csv('train.csv', sep='###__###', header=None, engine='python', encoding='UTF-8')
test = pd.read_csv('test.csv', sep='###__###', header=None, engine='python', encoding='UTF-8')
train.columns = ['ID', 'Age', 'Gender', 'Education', 'QueryList']
test.columns = ['ID', 'QueryList']
querys = train['QueryList'].apply(lambda x: x.split('\t')).to_frame()
print(querys)

net_pattern = r'(https?:\/\/)?[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+.+'
exclude_pattern = ''

# stopwords = []
# with open('stopWord.txt') as f:
#     for line in f.readlines():
#         stopwords.append(line.strip())
# print(stopwords)
#
# p_num = re.compile(r'\D+')
# after_dropnum_re = p_num.findall('9vfl4jk')

# 去掉停用词中的词，尽量先去掉所有停用词再继续分
# def processing_jieba(words_list):
#     for words in words_list:




#分词之后筛选
# def WashDataandJieBa(text, stopwords):
#     querys['jieba'] = querys['QueryList'].apply(processing_jieba)


# 划分train, validation,注意各个部分比例
# tf-idf? embedding后双向lstm?
# 缺失值有多少