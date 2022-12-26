import torch
from torch import nn


class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, bidirectional, num_classes, drop_prob=0.5):
        """
        初始化模型

        参数：
            vocab_size: int, 词表大小
            embed_size: int, 词向量维度
            num_hiddens: int, 隐藏层维度
            num_layers: int, 隐藏层层数
            bidirectional: bool, 是否双向
            num_classes: int, 类别数
            drop_prob: float, dropout概率

        """
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               dropout=drop_prob)
        self.decoder = nn.Linear(4 * num_hiddens, num_classes)
    
    def forward(self, inputs):
        """
        前向传播

        参数：
            inputs: Tensor, 输入
        返回：
            outputs: Tensor, 输出
        """
        inputs=inputs.permute(1,0)
        # torch.Size([64, 64])
        embeddings = self.embedding(inputs)
        # torch.Size([64, 64, 64])
        outputs, _ = self.encoder(embeddings)
        # torch.Size([64, 64, 128])
        encoding = torch.cat((outputs[0], outputs[-1]), -1)
        # torch.Size([64, 256])
        outs = self.decoder(encoding)
        # torch.Size([64, 2])
        return outs


def testModel():
    """
    测试模型
    """
    model = BiRNN(vocab_size=5000, embed_size=100, num_hiddens=100, num_layers=2, bidirectional=True, num_classes=2)
    print(model)


if __name__ == "__main__":
    testModel()
    