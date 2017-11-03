#!/usr/bin/env python
# pylint: disable=C0103
import numpy as np
import sklearn.linear_model as linear_model
from sklearn.metrics import mean_squared_error


def feature_with_complementation(i):
    """
    构造输入数字的二维特征向量,向量分量分别是数字对3,5和7的取余计算结果
        :param i: 输入的数字
        :returns: ndarray 构造的三维特征向量
    """
    return np.array([i % 3, i % 5, i % 7])


def feature_complementation_is_zero(i):
    """
    构造输入数字的二维特征向量,当输入的数字对3,5或者7的取余计算结果为0时,对应分量为1,否则分量为0
        :param i: 输入的数字
        :returns: ndarray 构造的三维特征向量
    """
    return np.array([1 if i % 3 == 0 else 0,
                     1 if i % 5 == 0 else 0,
                     1 if i % 7 == 0 else 0])


def construct_sample_label(i):
    """
    对预测的真值结果映射成数字,以便计算机进行处理
    """
    if i % (3 * 5 * 7) == 0:
        return np.array([7])
    elif i % 35 == 0:
        return np.array([6])
    elif i % 21 == 0:
        return np.array([5])
    elif i % 15 == 0:
        return np.array([4])
    elif i % 7 == 0:
        return np.array([3])
    elif i % 5 == 0:
        return np.array([2])
    elif i % 3 == 0:
        return np.array([1])
    else:
        return np.array([0])


def train_and_test(feature_func, label_func):
    """
    使用给定的特征函数以及真值映射函数构建训练集和测试集,并进行训练和测试
    计算误差, 对比与预测值和真实值
    """
    x_train = np.array([feature_func(i) for i in range(101, 201)])
    y_train = np.array([label_func(i) for i in range(101, 201)])

    x_test = np.array([feature_func(i) for i in range(1, 101)])
    y_test = np.array([label_func(i) for i in range(1, 101)])

    regr = linear_model.LinearRegression()
    regr.fit(x_train, y_train)

    y_pred = regr.predict(x_test)

    # print("预测值与真实值对比:\r\n", np.column_stack((y_pred, y_test)))

    print("Mean squared error (均方差-误差): %.2f"
          % mean_squared_error(y_pred, y_test))
    print("预测值中的最小值:", y_pred.min())
    print("预测值中的最大值:", y_pred.max())


feature_array = [("使用x对3,5,7的取余计算结果作为特征向量的分量",
                  feature_with_complementation),

                 ("判断x对3,5,7的取余计算结果是否为0,为0--->分量为1,不为0--->分量为0",
                  feature_complementation_is_zero)]

for tup in feature_array:
    print("-" * 40)
    print(tup[0])
    train_and_test(tup[1], construct_sample_label)
