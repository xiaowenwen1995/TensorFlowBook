# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:09:47 2018

@author: xiaowenwen
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#download dataset from mnist
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#learning_rate: learning rate: 0.01
#max_samples: the max number of samples: 400000
#batch_size: the size of batch: 128
#display_step: print result each 10 times
learning_rate = 0.01
max_samples = 400000
batch_size = 128
display_step = 10

#n_input: the width of input image: 28
#n_steps: unrolled steps of LSTM, the hight of input image: 28
#n_hidden: LSTM node: 256
#n_classes: the number of class
n_input = 28
n_steps = 28
n_hidden = 256
n_classes = 10

#x: input
#y: output
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

#softmax: w, b
weights = tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

#Bidirectional LSTM
def BiRNN(x, weights, biases):
    
    #(batch_size, n_steps, n_inputs) => n_steps (batch_size, n_input)
    #switch batch_size and n_steps
    #(n_steps * batch_size, n_input)
    #split: n_steps (batch_size, n_input)
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x, n_steps)
    
    #create forward and backward LSTM Cell
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    
    #create RNN
    #W*outputs+b
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32)
    return tf.matmul(outputs[-1], weights) + biases

pred = BiRNN(x, weights, biases)

#cost: mean
#optimizer: Adam
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

#calculate the accuracy
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#init
init = tf.global_variables_initializer()

#training
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < max_samples:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(batch_size, n_steps, n_input)
        sess.run(optimizer, feed_dict = {x: batch_x, y: batch_y})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict = {x: batch_x, y: batch_y})
            loss = sess.run(cost, feed_dict = {x: batch_x, y: batch_y})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    #predicting    
    test_len = 10000
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy", sess.run(accuracy, feed_dict = {x: test_data, y: test_label}))