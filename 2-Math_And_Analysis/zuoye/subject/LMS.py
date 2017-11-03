#!/usr/bin/env python
# pylint: disable=C0103
"""
手写数字识别
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class LMS():
    def active(self, x, function="sigmoid", derivative=False):
        if function == "sigmoid":
            if derivative == False:
                return 1 / (1 + np.exp(-x))
            else:
                return np.exp(-x) / (1 + np.exp(-x))**2
        elif function == "line":
            if derivative == False:
                return x
            else:
                return np.ones_like(x)

    def __init__(self, shape=[2, 1], init=1):
        self.shape = shape

        # 使用高斯分布随机初始化一个权值矩阵作为训练前的初始权值
        self.W = np.random.normal(loc=0.0, scale=0.1, size=shape) * init
        self.resv = self.W
        # 这个b也是使用高斯分布随机初始化的一个1*10的矩阵偏差向量
        self.b = np.random.normal(loc=0.0, scale=0.1, size=shape[1])

    def train(self, data, vali, eta, is_grad=False):
        """
        训练函数
            :param data: 训练样本
            :param vali: 期望输出
            :param eta:  学习速率
            :param is_grad=False: 
        """
        nu = np.dot(data, self.W) + np.tile(self.b, (np.shape(vali)[0], 1))

        para = (vali - self.active(nu,
                                   function="sigmoid")) * self.active(nu,
                                                                      function="sigmoid",
                                                                      derivative=True)
        x = data
        # 计算平均梯度
        self.grad_ave = np.dot(np.transpose(x), para) / len(data)

        # 调整权值
        self.W = np.add(self.W, eta * self.grad_ave)
        # 调整误差
        self.b = np.add(self.b, eta * np.average(para))

    def valid(self, data):
        return self.active(np.dot(data, self.W))


lms = LMS([784, 10])
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
for itr in range(200):
    # batch_xs 训练样本，batch_ys 期望输出，每次取60行
    batch_xs, batch_ys = mnist.train.next_batch(60)
    lms.train(batch_xs, batch_ys, 0.5)
    if(itr % 20 == 0):
        # 取测试集输入
        x = mnist.test.images
        # 取测试集希望输出
        label = mnist.test.labels
        # 预测
        y = lms.valid(x)
        # print(np.max(lms.W))
        cont = 0
        num = len(y)

        for itra, itrb in zip(y, label):
            la = np.where(itra == np.max(itra))[0][0]
            lb = np.where(itrb == np.max(itrb))[0][0]
            if(la == lb):
                cont += 1
        print("Itr:{} Acc:{}".format(itr, cont / num))

mpl.style.use('seaborn-darkgrid')
batch_xs, batch_ys = mnist.train.next_batch(300)
fig = plt.figure()
for itrx in range(3):
    for itry in range(3):
        for itmg, itlb in zip(batch_xs, batch_ys):
            la = np.where(itlb == np.max(itlb))[0][0]
            if(la == itrx * 3 + itry + 1):
                ax = fig.add_subplot(3, 3, la)
                ax.imshow(np.reshape(itmg, [28, 28]))
                plt.text(0, 0, "label:%d" % la)
                plt.xticks([])
                plt.yticks([])
                break
plt.matshow(np.reshape(lms.W[:, 2], [28, 28]), cmap=plt.get_cmap("PiYG"))
plt.show()
