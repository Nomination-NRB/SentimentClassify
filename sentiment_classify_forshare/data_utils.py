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

# nltk.download('punkt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def read_weibo(tag='train', data_root="data\\weibo_senti_100k"):
#     data = []
#     input_file = os.path.join(data_root, "{0}.csv".format(tag))
#     print("input_file: ",input_file)
#     flag=0
#     with open(input_file, 'r', encoding="utf8") as f:
#         for line in tqdm(f):
#             if flag==0:
#                 flag=1
#                 continue
#             line = line.strip()
#             i = 0
#             j = 0
#             for i in range(len(line)):
#                 if line[i] == ',':
#                     j+=1
#                 if j == 2:
#                     break
#             label = int(line[-1])
#             review = line[i+1:-2]
#             # print(label, review)
#             data.append([review, label])  # 评论文本字符串和01标签
#     random.shuffle(data)
#     return data

def read_weibo(tag='train', data_root="data\\weibo_senti_100k"):
    data = []
    input_file = os.path.join(data_root, "{0}.csv".format(tag))
    with open(input_file, 'r', encoding="utf8") as f:
        head_line = f.readline()
        for line in tqdm(f):
            line = line.strip()
            label = int(line[0])
            review = line[2:]
            data.append([review, label])  # 评论文本字符串和01标签
    random.shuffle(data)
    return data


def get_tokenized_weibo(data):  # 将每行数据的进行空格切割,保留每个的单词
    # 此处可以添加更复杂的过滤逻辑
    def tokenizer(text):
        return [tok for tok in jieba.cut(text)]

    return [tokenizer(review) for review, _ in data]


def get_vocab_weibo(data, min_count=1):
    '''
    @params:
        data: 同上
    @return: 数据集上的词典，Vocab 的实例（freqs, stoi, itos）
    '''
    tokenized_data = get_tokenized_weibo(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    # print(counter)
    # 统计所有的数据
    dict = vocab.vocab(counter, min_freq=min_count)  # 构建词汇表
    # 加入<pad> 和 <unk>
    dict.insert_token("<pad>", 0)
    dict.insert_token("<unk>", 1)
    return dict


def preprocess_weibo(data, vocab, max_l=64):
    def pad(x):  # 填充
        return x[:max_l] if len(x) > max_l else x + [vocab["<pad>"]] * (max_l - len(x))

    tokenized_data = get_tokenized_weibo(data)
    # pprint(tokenized_data[:10])
    padded_tokenized_data = []
    for words in tokenized_data:
        indexed_words = [vocab[word] if word in vocab else vocab["<unk>"] for word in words]
        padded_words = pad(indexed_words)
        padded_tokenized_data.append(padded_words)
    # pprint(padded_tokenized_data[:10])
    features = torch.tensor(padded_tokenized_data)
    labels = torch.tensor([score for _, score in data])
    return features, labels

def make_weibo_dataset(batch_size=64, max_length=64, min_count=5):
    # 读取文本数据
    train_data, test_data = read_weibo(tag="train"), read_weibo(tag="test")

    # 获取字典
    vocab = get_vocab_weibo(train_data, min_count)
    # *号语法糖,解绑参数，获取dataset对象
    train_set = Data.TensorDataset(*preprocess_weibo(train_data, vocab, max_length))
    test_set = Data.TensorDataset(*preprocess_weibo(test_data, vocab, max_length)) 
    # 获取迭代器
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_set, batch_size)

    return train_iter, test_iter, vocab


# def make_weibo_testset(file_path, vocab_path, batch_size=64, max_length=64):
#     # 读取数据
#     data = []
#     flag=0
#     with open(file_path, 'r', encoding="utf8") as f:
#         for line in f:
#             if flag==0:
#                 flag=1
#                 continue
#             review = line.strip()
#             label = 0  # 标签固定为0
#             data.append([review, label])
#
#     # 读入词典
#     vocab = read_vocab(vocab_path)
#
#     # *号语法糖,解绑参数，获取dataset对象
#     data_set = Data.TensorDataset(*preprocess_weibo(data, vocab, max_length))  # 相当于将函数参数是函数结果
#     # 获取迭代器
#     data_iter = Data.DataLoader(data_set, batch_size)
#
#     return data_iter, vocab


def save_vocab(vocab, path):
    # print(vocab.get_itos())
    with open(path, 'w', encoding="utf8") as output:
        print("\n".join(vocab.get_itos()), file=output)


def read_vocab(vocab_path):
    vocab_dict = {}
    with open(vocab_path, 'r', encoding="utf8") as f:
        for line in f:
            word = line[:-1]
            # print("*{0}*".format(word))
            if word == "": continue
            vocab_dict[word] = 1
    dict = torchtext.vocab.vocab(vocab_dict, min_freq=0)

    return dict


def load_pretrained_embedding(words, pretrained_vocab_path=None, emb_size=100, type="glove"):
    '''
    @params:
        words: 需要加载词向量的词语列表，以 itos (index to string) 的词典形式给出
        pretrained_vocab: 预训练词向量
        type: 词向量的种类
    @return:
        embed: 加载到的词向量
    '''
    # embed = torch.zeros(len(words), emb_size)  # 初始化为len*100维度
    embed = torch.normal(mean=0, std=1, size=(len(words), emb_size))

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


def test(argv):
    train_data, test_data = read_weibo(tag="train"), read_weibo(tag="test")
    train_iter, test_iter, _ = make_weibo_dataset(min_count=5)
    for X, y in train_iter:
        print('X', X, 'y', y)
        break
    # print('#batches:', len(train_iter))
    # for X, y in test_iter:
    #     print('X', X, 'y', y)
    #     break

    # cache_dir = "/home/kesci/input/GloVe6B5429"
    # glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=cache_dir)
    #
    # net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, glove_vocab))
    # net.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它


# 测试相关函数用
if __name__ == '__main__':
    test(sys.argv)
