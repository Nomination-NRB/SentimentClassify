import time
import torch
import utils
from model import BiRNN
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy(data_iter, model, device):
    """
    计算准确率

    参数：
        data_iter: DataLoader, 数据迭代器
        model: 模型
        device: 设备
    返回：
        acc: float, 准确率
    """
    model.eval()
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            acc_sum += (model(X).argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]
    return acc_sum / n


def train(train_loader, test_loader, model, loss, optimizer, device, epochs):
    model = model.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        model.train()
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = accuracy(test_loader, model, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


if __name__ == '__main__':
    train_loader, test_loader, vocab_dict = utils.makeDataset()
    utils.saveVocab(vocab_dict)

    embed_size = 64
    num_hiddens = 64
    num_layers = 2
    bidirectional = True
    num_classes = 2
    drop_prob = 0.9
    lr = 0.01
    epochs = 10

    net = BiRNN(vocab_size=len(vocab_dict), embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers,
                bidirectional=bidirectional, num_classes=num_classes, drop_prob=drop_prob)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    train(train_loader, test_loader, net, loss, optimizer, device, epochs)
    outputModelPath = 'output/model.pt'
    torch.save(net, outputModelPath)
