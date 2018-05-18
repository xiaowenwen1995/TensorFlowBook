# -*- coding: utf-8 -*-
"""
Created on Thu May 10 20:44:50 2018

@author: xiaowenwen
"""

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))