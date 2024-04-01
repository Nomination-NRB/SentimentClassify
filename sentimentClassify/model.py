import torch
import torch.nn as nn
import torch.nn.functional as F


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


def testModelBiRNN():
    """
    测试模型
    """
    model = BiRNN(vocab_size=5000, embed_size=100, num_hiddens=100, num_layers=2, bidirectional=True, num_classes=2)
    print(model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, num_classes, drop_prob=0.5):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(drop_prob)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, attention_mask=None):
        embeddings = self.embedding(inputs)
        embeddings = self.pos_encoder(embeddings)
        transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=attention_mask)
        # 取 transformer_output 的最后一层作为 pooled_output
        pooled_output = transformer_output[-1, :, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.decoder(pooled_output)
        return logits



def testTransformerModel():
    """
    测试Transformer模型
    """
    model = TransformerClassifier(vocab_size=5000, embed_size=100, num_heads=4, num_layers=2, num_classes=2)
    print(model)


if __name__ == "__main__":
    testTransformerModel()
    