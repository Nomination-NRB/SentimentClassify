# -*- coding:utf-8 -*-

import os
import sys
import time

import torch
from torch import nn

from models.BiRNN import BiRNN
from data_utils import make_weibo_dataset, load_pretrained_embedding, save_vocab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_accuracy(data_iter, net, mydevice=None):
    if mydevice is None and isinstance(net, torch.nn.Module):
        mydevice = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                acc_sum += (net(X.to(mydevice)).argmax(dim=1) == y.to(mydevice)).float().sum().cpu().item()
                net.train()
            else:
                if 'is_training' in net.__code__.co_varnames:
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train(train_iter, test_iter, net, loss, optimizer, mydevice, num_epochs):
    net = net.to(mydevice)
    print("training on ", mydevice)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(mydevice)
            y = y.to(mydevice)
            # print(X)
            # print(y)
            y_hat = net(X)
            # print(y_hat)
            l = loss(y_hat, y)  # 交叉熵损失函数
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  # 优化方法
            train_l_sum += l.cpu().item()  # 进入cpu中统计
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def main(argv):
    train_iter, test_iter, vocab = make_weibo_dataset(max_length=32, min_count=5)
    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break
    print('#batches:', len(train_iter))  # 391个批次,每个批次64个样本

    # 保存词典
    output_vocab_path = "output\\model.vocab"
    save_vocab(vocab, output_vocab_path)
    print("#save vocab: ", output_vocab_path)

    # 定义模型
    embed_size, num_hiddens, num_layers, prob = 64, 64, 2, 0.9
    net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers, prob)
    print("#model:", net)

    # 读入预训练词嵌入模型
    print("#loading pretrained embedding...")
    net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.get_itos(), emb_size=embed_size, type="glove"))

    # 定义优化器
    lr, num_epochs = 0.001, 10
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    print("#opitmizer: ", optimizer)
    loss = nn.CrossEntropyLoss()
    print("#loss: ", loss)

    # 执行训练过程
    train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

    # 保存模型
    output_model_path = "output\\model.pt"
    torch.save(net, output_model_path)
    print("#save model: ", output_model_path)


if __name__ == '__main__':
    main(sys.argv)
