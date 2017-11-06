import tensorflow as tf
import numpy as np

A = tf.placeholder(dtype=tf.float64, shape=[2, 2])
b = tf.placeholder(dtype=tf.float64, shape=[2])
#矩阵函数的使用
A_pow = tf.sin(A)
A_relu = tf.nn.relu(A) #relu是一个函数：当x<=0时relu(x)=0,当x>0时relu(x)=x

#矩阵求逆
A_inverse = tf.matrix_inverse(A)

#矩阵转置
A_T = tf.transpose(A)
#生成对角矩阵
b_diag = tf.diag(b) # b是一个一维数组，已b中的元素作为对角矩阵中对角线上的元素
# 生成单位矩阵
I = tf.eye(6)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(sess.run(A_pow, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))


print(sess.run(A_relu, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))

print(sess.run(A_inverse, 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))

print(sess.run([b_diag, I], 
               feed_dict={A: [[1, 2], [-1, 1]], 
                          b: [1, 1]}))