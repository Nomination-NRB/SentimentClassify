import os
import random
import jieba
import torch
import collections
import torch.utils.data as Data
import torchtext.vocab as vo
from tqdm import tqdm


def readData(tag='train', data_root="./data/"):
    """
    读取数据

    参数：
        tag: train, test
        data_root: 数据集根目录
    返回：
        data: list, 元素为[review, label]，review为字符串，label为0或1
    """
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


def tokenizer(text):
    return [tok for tok in jieba.cut(text)]


def getTokenized(data):
    """
    获得分词后的数据

    参数：
        data: list, 元素为[review, label]，review为字符串，label为0或1
    返回：
        data: list, 元素为[review, label]，review为分词后的list，label为0或1
    """
    return [tokenizer(review) for review, _ in data]


def getVocab(tokendata):
    """
    获得词表

    参数：
        data: list, 元素为[review, label]，review为分词后的list，label为0或1
    返回：
        vocab: list, 词表
    """
    counter = collections.Counter([tk for st in tokendata for tk in st])
    vocabDict = vo.vocab(counter, min_freq=1)
    vocabDict.insert_token('<unk>', 1)
    vocabDict.insert_token('<pad>', 0)
    return vocabDict


def pad(text, vocab, maxLen=64):
    return text[:maxLen] if len(text) > maxLen else text + [vocab['<pad>']] * (maxLen - len(text))


def preProcessData(tokendata, data, vocab, maxLen=64):
    """
    预处理数据

    参数：
        data: list, 元素为[review, label]，review为分词后的list，label为0或1
        vocab: list, 词表
        maxLen: int, 最大长度
    返回：
        data: list, 元素为[review, label]，review为分词后的list，label为0或1
    """

    padTokenData = []
    for words in tokendata:
        indexWord = [vocab[words] if words in vocab else vocab['<unk>'] for words in words]
        padWord = pad(indexWord, vocab, maxLen)
        padTokenData.append(padWord)
    features = torch.tensor(padTokenData)
    labels = torch.tensor([label for _, label in data])
    return features, labels


def makeDataset(batchsize=64, maxlength=64, mincount=5):
    """
    生成数据集

    参数：
        batchsize: int, 批大小
        maxlength: int, 最大长度
        mincount: int, 最小词频
    返回：
        train_iter: 训练集迭代器
        test_iter: 测试集迭代器
        vocab: 词表
    """
    trainData = readData(tag='train')
    testData = readData(tag='test')
    trainTokenData = getTokenized(trainData)
    testTokenData = getTokenized(testData)
    vocab = getVocab(trainTokenData)
    trainFeatures, trainLabels = preProcessData(trainTokenData, trainData, vocab, maxlength)
    testFeatures, testLabels = preProcessData(testTokenData, testData, vocab, maxlength)
    train_dataset = Data.TensorDataset(trainFeatures, trainLabels)
    test_dataset = Data.TensorDataset(testFeatures, testLabels)
    train_loader = Data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = Data.DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
    return train_loader, test_loader, vocab


def saveVocab(vocab, path='./output/model.vocab'):
    """
    保存词表

    参数：
        vocab: 词表
        path: 保存路径
    """
    with open(path, 'w', encoding="utf8") as output:
        print("\n".join(vocab.get_itos()), file=output)


def loadVocab(path='./output/model.vocab'):
    """
    读取词表

    参数：
        path: 读取路径
    返回：
        vocab: 词表
    """
    vocab = {}
    with open(path, 'r', encoding="utf8") as f:
        for i, line in enumerate(f):
            vocab[line.strip()] = i
    vocabDict = vo.vocab(vocab, min_freq=0)
    return vocabDict


def loadPreTrainEmbedding(words, preTrainVocabPath='./data/glove/', embedSize=100, type='glove'):
    """
    加载预训练词向量

    参数：
        words: 词表
        preTrainVocabPath: 预训练词向量路径
        embedSize: 词向量维度
    返回：
        embed: 词向量
    """
    embed = torch.normal(mean=0, std=1, size=(len(words), embedSize))
    if type == 'glove':
        preTrainVocab = vocab.GloVe(name='6B', dim=embedSize, cache=preTrainVocabPath)
    elif type == 'word2vec':
        preTrainVocab = vocab.Word2VecTextFile(preTrainVocabPath)
    else:
        return embed
    preTrainVocabSize = preTrainVocab.vectors.shape[0]
    outofvocab = 0
    for i, word in enumerate(words):
        try:
            index = preTrainVocab.stoi[word]
            if preTrainVocabSize == embedSize:
                embed[i, :] = preTrainVocab.vectors[index]
            elif preTrainVocabSize > embedSize:
                embed[i, :] = preTrainVocab.vectors[index][:embedSize]
            else:
                embed[i, :] = preTrainVocab.vectors[index] + [0] * (embedSize - preTrainVocabSize)
        except:
            outofvocab += 1
    if outofvocab > 0:
        print('out of vocab: %d' % outofvocab)
    return embed


if __name__ == "__main__":
    # test the function you want
    train_loader, test_loader, vocab = makeDataset()
