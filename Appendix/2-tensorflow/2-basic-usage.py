#!/use/bin/env python
# pylint: disable=C0103

"""
tensorflow 的基本用法
"""

import tensorflow as tf


# 创建一个常量op,返回一个1*2的矩阵
matrix1 = tf.constant([[3., 3.]])
# 创建一个常量op,返回一个2*1的矩阵
matrix2 = tf.constant([[2.], [2.]])

# 创建一个矩阵乘法的op,返回matrix1和matrix2的矩阵乘法结果
product = tf.matmul(matrix1, matrix2)

# 创建一个会话,启动tensorflow的默认图
sess = tf.Session()
# 执行矩阵乘法op,会话会自动传入所需的输入,返回矩阵乘法的结果
result = sess.run(product)
print(result)

sess.close()


# 创建一个变量,初始值为0
state = tf.Variable(0, name="counter")

one = tf.constant(1)
new_value = tf.add(state, one)
udpate = tf.assign(state, new_value)

# 启动图之后,变量必须经过'初始化'(init)op初始化
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # 运行init_op
    sess.run(init_op)

    # 打印state的初始值
    print(sess.run(state))

    for _ in range(3):
        sess.run(udpate)
        print(sess.run(state))


# 同时获取多个tensor
input1 = tf.constant(3.)
input2 = tf.constant(2.)
input3 = tf.constant(5.)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as sess:
    result = sess.run([mul, intermed])
    print (result)
