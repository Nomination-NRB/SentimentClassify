# 情感分类

## 简介

对句子进行情感分类

## 技术栈

深度学习：BiRNN->LSTM训练模型

## 目录

```
sentimentClassify
├─ data
│    ├─ infer.txt
│    ├─ test.csv
│    └─ train.csv
├─ inference.py
├─ model.py
├─ output
│    ├─ model.pt
│    └─ model.vocab
├─ total.ipynb
├─ train.py
└─ utils.py
```

## 使用方法

```bash
git clone https://github.com/Nomination-NRB/SentimentClasssify
```

在vscode或者其他编译器打开项目文件夹

激活本项目具体使用的环境，切换到SentimentClasssify/requirements.txt目录下在终端执行该命令即可

```bash
pip install -r requirements.txt
```

1. 运行train.py将使用data文件夹下的训练数据与测试数据
2. 运行inference.py将使用刚刚训练的模型进行情感推理



## 文件说明

- inference.py
  - 用于推理data/infer.txt的内容
  - 训练完成后，运行此文件可以使用刚刚训练保存的模型来推理
- model.py
  - 模型的定义与测试
- train.py
  - 训练过程
- utils.py
  - 处理评论所需的功能函数
- total.ipynb
  - 将所有部分整合到一起的文件，适用于Google colab运行
