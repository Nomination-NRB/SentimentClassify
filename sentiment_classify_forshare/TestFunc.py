# -*- coding:utf-8 -*-

import collections
import os
import random
import sys

import jieba
import torch
import torch.utils.data as Data
import torchtext.vocab as vocab
from tqdm import tqdm

import nltk
import re


def read_weibo(tag='training-processed', data_root="data\\weibo_senti_100k"):
    data = []
    input_file = os.path.join(data_root, "{0}.csv".format(tag))
    with open(input_file, 'r', encoding="utf8") as f:
        for line in tqdm(f):
            line = line.strip()
            # print("line: ",line)
            # print(len(line))
            i = 0
            j = 0
            for i in range(len(line)):
                if line[i] == ',':
                    j += 1
                if j == 5:
                    break
            if int(line[0]) == 4:
                label = 1
            else:
                label = int(line[0])
            # print(i)
            review = line[i + 1:-5]
            # print(review)
            data.append([review, label])  # 评论文本字符串和01标签
    # random.shuffle(data)
    # print(data[:10])
    return data


def get_tokenized_weibo(data):  # 将每行数据的进行空格切割,保留每个的单词
    # 此处可以添加更复杂的过滤逻辑
    def tokenizer(text):
        return [tok for tok in jieba.cut(text)]

    mylist = [tokenizer(review) for review, _ in data]
    # print(mylist[:10])
    return mylist


# nltk.download('punkt')
def get_vocab_weibo(data, min_count=1):
    """
    @params:
        data: 同上
    @return: 数据集上的词典，Vocab 的实例（freqs, stoi, itos）
    """
    tokenized_data = get_tokenized_weibo(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # print(counter)
    # 统计所有的数据
    dict2 = vocab.vocab(counter, min_freq=min_count)  # 构建词汇表
    # 加入<pad> 和 <unk>
    dict2.insert_token("<pad>", 0)
    dict2.insert_token("<unk>", 1)
    return dict2


def load_pretrained_embedding(words, pretrained_vocab_path=None, emb_size=100, type="glove"):
    """
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
        type: 词向量的种类
    @return:
        embed: 加载到的词向量
    """
    # embed = torch.zeros(len(words), emb_size)  # 初始化为len*100维度
    print(words[:10])
    print(len(words))
    print(emb_size)
    embed = torch.normal(mean=0, std=1, size=(len(words), emb_size))
    print(embed.size())
    if type == "glove":
        # 先硬编码使用100d的glove向量
        pretrained_vocab = vocab.GloVe(name="6B", dim=100, cache="data\\glove")
    else:
        return embed

    pretrained_emb_size = pretrained_vocab.vectors[0].shape[0]
    oov_count = 0  # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            if pretrained_emb_size == emb_size:
                embed[i, :] = pretrained_vocab.vectors[idx]  # 将每个词语用训练的语言模型理解
            elif pretrained_emb_size < emb_size:
                embed[1, :] = pretrained_vocab.vectors[idx] + [0] * (emb_size - pretrained_emb_size)
            else:
                embed[1, :] = pretrained_vocab.vectors[idx][:emb_size]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    # print(embed.shape),在词典中寻找相匹配的词向量
    return embed


cache_dir = "data\\glove"
glove_vocab = vocab.GloVe(name='6B', dim=100, cache=cache_dir)
train_data, test_data = read_weibo(tag="test_sample"), read_weibo(tag="test_sample")
dict3 = get_vocab_weibo(train_data[:5], 1)
emb = load_pretrained_embedding(dict3.get_itos(), glove_vocab)
print(emb.size())
print(emb[0].size())
print(len(emb[0]))
print(emb.shape)
