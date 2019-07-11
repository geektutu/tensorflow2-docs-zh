---
title: TensorFlow 2 / 2.0 中文文档
date: 2019-07-09 00:10:10
description: TensorFlow 2 官方文档中文版，有删改，序言，介绍整个文档的构成。
tags:
- TensorFlow 2
- 官方文档
categories:
- TensorFlow2文档
image: post/tf2doc/tf.jpg
github: https://github.com/geektutu/tensorflow2-docs-zh
---

![TensorFlow 2.0](tf2doc/tf.jpg)

## 文档地址

- 文档地址：[TensorFlow 2 / 2.0 中文文档](https://geektutu.com/post/tf2doc.html)
- Github：[Github - tensorflow2-docs](https://github.com/geektutu/tensorflow2-docs-zh)
- 知乎专栏：[Zhihu - Tensorflow2-docs](https://zhuanlan.zhihu.com/geektutu)

## 目录(持续更新)

### 基础 - 机器学习基础 ML basics

1. [图像分类 Classify images](https://geektutu.com/post/tf2doc-ml-basic-image.html)
2. [文本分类 Classify text](https://geektutu.com/post/tf2doc-ml-basic-text.html)
3. [结构化数据分类 Classify structured data](https://geektutu.com/post/tf2doc-ml-basic-structured-data.html)
4. [回归 Regression](https://geektutu.com/post/tf2doc-ml-basic-regression.html)
5. [过拟合与欠拟合 Overfitting and underfitting](https://geektutu.com/post/tf2doc-ml-basic-overfit.html)
6. 保存和恢复模型 Save and restore models

### 进阶 - 自定义

1. 张量和操作 Tensors and operations
2. 自定义层 Custom layers
3. 自动微分 Automatic differentiation
4. 自定义训练：攻略 Custom training：walkthrough
5. 动态图机制 TF function and AutoGraph

## 极客兔兔实战

### 监督学习

1. [mnist手写数字识别(CNN卷积神经网络)](https://geektutu.com/post/tensorflow2-mnist-cnn.html)
2. [监督学习玩转 OpenAI gym game](https://geektutu.com/post/tensorflow2-gym-nn.html)

### 强化学习

1. [强化学习 Q-Learning 玩转 OpenAI gym](https://geektutu.com/post/tensorflow2-gym-q-learning.html)
2. [强化学习 DQN 玩转 gym Mountain Car](https://geektutu.com/post/tensorflow2-gym-dqn.html)
3. [强化学习 70行代码实战 Policy Gradient](https://geektutu.com/post/tensorflow2-gym-pg.html)

## 声明

**TensorFlow 2 中文文档**主要参考 [TensorFlow官网](https://www.tensorflow.org/beta/tutorials/keras)，书写而成。选取了一些有价值的章节作总结，内容目录基本与官方文档一致，但在内容上作了大量的简化，以代码实践为主。TensorFlow 是机器学习的高阶框架，功能强大，接口很多，TensorFlow 2 废弃了大量重复的接口，将 Keras 作为搭建网络的主力接口，也添加了很多新的特性，极大地改进了可用性，能有效地减少代码量。

**TensorFlow 2 中文文档**的目的是选取官方文档中有代表性的内容，帮助大家快速入门，一览TensorFlow 在图像识别、文本分类、结构化数据等方面的风采。介绍 TensorFlow 1.x 的文档已经很多，所以这份文档侧重于总结 TensorFlow 2 的新特性。

TensorFlow官网的文档遵循[署名 4.0 国际 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.zh)协议，代码遵循[Apache 2.0 协议](https://www.apache.org/licenses/LICENSE-2.0)，本文档完全遵守上述协议。将在显著地方注明来源。

代码基于**Python3**和**TensorFlow 2.0 beta**实现。

力求简洁，部分代码删改过，例如兼容Python 2.x的代码均被删除。