import re
import torch
import torch.utils.data as Data
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cleanText(text):
    """
    清洗文本

    参数：
        text: str, 文本
    返回：
        text: str, 清洗后的文本
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def makeIter(texts, vocab_path='output/model.vocab', batch_size=64, max_length=64):
    """
    生成迭代器

    参数：
        texts: list, 文本列表
        vocab_path: str, 词表路径
        batch_size: int, 批大小
        max_length: int, 最大长度
    返回：
        data_iter: DataLoader, 数据迭代器
    """
    data = []
    for text in texts:
        text = cleanText(text)
        data.append([text, -1])
    vocab = utils.loadVocab(vocab_path)
    tokenizedData = utils.getTokenized(data)
    features, labels = utils.preProcessData(tokenizedData, data, vocab, max_length)
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=False)
    return data_iter


def makeIterOfFile(filePath='data/infer.txt', vocab_path='output/model.vocab', batch_size=64, max_length=64):
    """
    生成迭代器

    参数：
        filePath: str, 文件路径
        vocab_path: str, 词表路径
        batch_size: int, 批大小
        max_length: int, 最大长度
    返回：
        data_iter: DataLoader, 数据迭代器
    """
    data = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            line = cleanText(line)
            line = line.strip()
            if line:
                data.append([line, -1])

    vocab = utils.loadVocab(vocab_path)
    tokenizedData = utils.getTokenized(data)
    features, labels = utils.preProcessData(tokenizedData, data, vocab, max_length)
    dataset = Data.TensorDataset(features, labels)
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=False)
    return data_iter, vocab


def inference(text_iter, device, outputModelPath='output/model.pt'):
    """
    预测

    参数：
        text: str, 文本
        outputModelPath: str, 模型路径
    返回：
        label: int, 类别
    """
    model = torch.load(outputModelPath)
    model.eval()
    model = model.to(device)
    result = []
    with torch.no_grad():
        for X, y in text_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            temp = y_hat.argmax(dim=1).cpu().numpy()
            result.extend(temp)
    return result


if __name__ == '__main__':
    text_iter, vocab = makeIterOfFile()
    print("#vocab: ", len(vocab))
    print('#batches:', len(text_iter))
    result = inference(text_iter, device)
    print(result)
