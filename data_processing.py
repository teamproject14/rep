import pandas as pd
import jieba
import re
train = pd.read_csv('train.csv', sep='###__###', header=None, engine='python', encoding='UTF-8')
test = pd.read_csv('test.csv', sep='###__###', header=None, engine='python', encoding='UTF-8')
train.columns = ['ID', 'Age', 'Gender', 'Education', 'QueryList']
test.columns = ['ID', 'QueryList']
data = train.iloc[:,:-1]
querys = train['QueryList'].apply(lambda x: x.split('\t'))
print(querys)

# 包括网址、IP地址、数字、版本号
net_pattern = r'(https?:\/\/)?[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)+.+'
p_net = re.compile(net_pattern)

# 排除数字、版本号
exclude_pattern1 = r'\d+(\.\d+){1,2}[^.]'
exclude_pattern2 = r'\d+(\.\d+){1,2}$'
p_exclude1 = re.compile(exclude_pattern1)
p_exclude2 = re.compile(exclude_pattern2)

# 把包括在exclude_pattern里的IP地址找回
IP_pattern1 = r'\d+(\.\d+){3}[^.]'
IP_pattern2 = r'\d+(\.\d+){3}$'
p_IP1 = re.compile(IP_pattern1)
p_IP2 = re.compile(IP_pattern2)

def count_net(l):
    count = 0
    for words in l:
        if p_net.match(words):
            if p_exclude1.match(words) or p_exclude2.match(words):
                if p_IP1.match(words) or p_IP2.match(words):
                    count += 1
            else:
                count += 1
    return count

data['net_count'] = querys.apply(count_net)

eng_pattern1 = r'[a-zA-Z]{2}'
eng_pattern2 = r'[a-zA-Z]{3,}'
p_eng1 = re.compile(eng_pattern1)
p_eng2 = re.compile(eng_pattern2)

def count_eng1(l):
    count = 0
    for words in l:
        if p_eng1.match(words):
            count += 1
    return count

def count_eng2(l):
    count = 0
    for words in l:
        if p_eng2.match(words):
            count += 1
    return count

data['eng_count1'] = querys.apply(count_eng1)
data['eng_count2'] = querys.apply(count_eng2)

age_group = data.groupby('Age').mean().loc[:, ['net_count', 'eng_count1', 'eng_count2']]
gender_group = data.groupby('Gender').mean().loc[:, ['net_count', 'eng_count1', 'eng_count2']]
education_group = data.groupby('Education').mean().loc[:, ['net_count', 'eng_count1', 'eng_count2']]




age0_raio = sum(data['Age']==0)/data.shape[0] #0.01666
gender0_ratio = sum(data['Gender']==0)/data.shape[0] # 0.02155
education0_ratio = sum(data['Education']==0)/data.shape[0] # 0.0928

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