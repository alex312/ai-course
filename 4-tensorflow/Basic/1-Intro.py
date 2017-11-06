#tensorflow库引入
import tensorflow as tf
import numpy as np


a = tf.constant(np.array([1,2]))
with tf.Session() as sess:
    print(sess.run(a))