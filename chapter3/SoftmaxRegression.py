#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 16:39:04 2018

@author: xiaowenwen
"""

"""
基本流程
1、定义算法公式
2、定义loss，选定优化器，并指定优化器优化loss
3、迭代地对数据进行训练
4、在测试集或者验证集上对准确率进行测评
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

#download dataset from mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

#register session
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#softmax regression: y = softmax(Wx+b)
y = tf.nn.softmax(tf.matmul(x, w) + b)

#loss function: cross-entropy 交叉熵
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#optimize function: stochastic gradient descent 随机梯度下降
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#run
tf.global_variables_initializer().run()

#training
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

#predicting    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
