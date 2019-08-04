---
title: TensorFlow 2 中文文档 - IMDB 文本分类
date: 2019-07-09 00:40:10
description: TensorFlow2.0 TF2.0 TensorFlow 2 / 2.0 官方文档中文版，文本分类 Classify text，示例使用 IMDB 数据集。
tags:
- TensorFlow 2
- 官方文档
keywords:
- Classify text
- TensorFlow2.0
- IMDB datasets
- 文本分类
nav: TensorFlow
categories:
- TensorFlow2文档
image: post/tf2doc-ml-basic-text/imdb-sm.jpg
github: https://github.com/geektutu/tensorflow2-docs-zh
---

**TF2.0 TensorFlow 2 / 2.0 中文文档 - 文本分类 Classify text**

主要内容：使用迁移学习算法解决一个典型的二分分类(Binary Classification)问题——电影正向评论和负向评论分类。

这篇文档使用包含有50,000条电影评论的 IMDB 数据集，25,000用于训练，25,000用于测试。而且训练集和测试集是均衡的，即其中包含同等数量的正向评论和负向评论。

代码使用`tf.keras`和`TensorFlow Hub`，TensorFlow Hub 是一个用于迁移学习的平台/库。

```python
import numpy as np
import tensorflow as tf # 2.0.0-beta1
import tensorflow_hub as hub # 0.5.0
import tensorflow_datasets as tfds
```

## 下载 IMDB 数据集

![IMDB datasets](tf2doc-ml-basic-text/imdb.jpg)

IMDB 数据集在`tfds`中是可以直接获取的，调用时会自动下载到你的机器上。

```python
# 进一步划分训练集。
# 60%(15,000)用于训练，40%(10,000)用于验证(validation)。
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)
```

## 数据格式

每个例子包含一句电影评论和对应的标签，0或1。0代表负向评论，1代表正向评论。

看一下前十条数据。

```python
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch
```

```python
<tf.Tensor: id=220, shape=(10,), dtype=string, numpy=
array([b"As a lifelong fan of Dickens, I have ...",
       b"Oh yeah! Jenna Jameson did it again! ...",
       b"I saw this film on True Movies (which ...",
       b'This was a wonderfully clever and ...',
       b'I have no idea what the other reviewer ...',
       b"This was soul-provoking! I am an ...",
       b'Just because someone is under ...',
       b'A very close and sharp discription ...',
       b"This is the most depressing film ..."],
      dtype=object)>
```

前十个标签。

```python
train_labels_batch
# <tf.Tensor: id=221, shape=(10,), dtype=int64, numpy=array([1, 1, 1, 1, 1, 1, 0, 1, 1, 0])>
```

## 搭建模型

神经网络需要堆叠多层，架构上需要考虑三点。

- 文本怎么表示？
- 模型需要多少层？
- 每一层多少个_隐藏节点_

一种表示文本的方式是将句子映射为向量(embeddings vectors)，或者称为文本嵌入(text embedding)。嵌入方法很多，比如我们可以采用最简单的独热编码，假设常用单词总共1000个，给每一个单词一个独热编码。假设每句话由10个单词构成，那么每句话均可以映射到10x1000的二维空间中。那么某句话就可以表示为：

```python
# 10x1000的二维向量表示一句话
[[0, 0, 0, 1, 0, ... 0],
 [0, 0, ..., 0, 1,... 0],
 [0, 0, 0, 0, 0, ... 1],
 ... 
 [0, 0, 0, 0, 0, ... 1]]
```

文本嵌入的方法很多，要考虑的因素也很多，比如同义词如何处理，维度过高怎么办？知乎上有比较详细的回答：[word embedding的解释](https://www.zhihu.com/question/32275069)。

我们可以使用一个预训练(pre-trained)好的文本嵌入模型作为第一层，有3个好处。

- 不用担心文本处理。
- 能从迁移学习中受益。
- 嵌入后size固定，处理起来简单。

接下来从 TensorFlow Hub 中选用的**pre-trained 文本嵌入模型**称为[google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)。

接下来创建一个 Keras Layer 使用这个模型将句子转为向量。取前三条评论试一试。注意无论句子的长度如何，最终的嵌入结果均为长度20的一维向量。

```python
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
```

```bash
<tf.Tensor: id=402, shape=(3, 20), dtype=float32, numpy=
array([[ 3.9819887, -4.4838037 , 5.177359, ... ],
       [ 3.4232912, -4.230874 , 4.1488533, ... ],
       [ 3.8508697, -5.003031 , 4.8700504, ... ]],
      dtype=float32)>
```

接下来，搭建完整的神经网络模型。

```python
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()
```

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param 
=================================================================
keras_layer (KerasLayer)     (None, 20)                400020    
_________________________________________________________________
dense (Dense)                (None, 16)                336       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 400,373
Trainable params: 400,373
Non-trainable params: 0
_________________________________________________________________
```

1. 第一层是 TensorFlow Hub 层，将句子转换为 tokens，然后映射每个 token，并组合成最终的向量。输出的维度是：句子个数 * 嵌入维度(20)。
2. 接下来是全连接层(Full-connected, FC)，即`Dense`层，16个节点。
3. 最后一层，也是全连接层，只有一个节点。使用`sigmoid`激活函数，输出值是float，范围0-1，代表可能性/置信度。

## 损失函数和优化器

`binary_crossentropy`更适合处理概率问题，`mean_squared_error`适合处理回归(Regression)问题。

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
```

## 训练模型

共 20 epochs，每个batch 512个数据。即对所有训练数据进行20轮迭代。在训练过程中，将监视模型在包含10,000条数据的验证集上的损失(loss)和正确率(accuracy)。

```python
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
```

```bash
Epoch 1/20
30/30 [========] - 6s 190ms/step - loss: 1.0201 - accuracy: 0.4331 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 2/20
30/30 [========] - 5s 159ms/step - loss: 0.7801 - accuracy: 0.4677 - val_loss: 0.7407 - val_accuracy: 0.5009
......
Epoch 20/20
30/30 [========] - 5s 152ms/step - loss: 0.1917 - accuracy: 0.9348 - val_loss: 0.2930 - val_accuracy: 0.8784
```

## 评估模型

`evaluate`返回2个值，Loss(误差，越小越好) 和 accuracy。

```python
results = model.evaluate(test_data.batch(512), verbose=0)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
# loss: 0.314
# accuracy: 0.866
```

这个非常基础的模型达到了87%的正确率，复杂一点的模型可以达到95%。

返回[文档首页](https://geektutu.com/post/tf2doc.html)

> 参考地址：[Text classification of movie reviews with Keras and TensorFlow Hub](https://www.tensorflow.org/beta/tutorials/keras/basic_text_classification_with_tfhub)

