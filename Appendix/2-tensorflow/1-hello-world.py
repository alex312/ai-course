#!/usr/bin/env python
# pylint: disable=C0103

"""
'Hello World' with tensorflow
"""

import tensorflow as tf

hello = tf.constant("Hello, tensorflow!")
sess = tf.Session()
helloResult = sess.run(hello)
print(helloResult)

a = tf.constant(10)
b = tf.constant(42)
result = sess.run(a + b)
print(result)

sess.close()
