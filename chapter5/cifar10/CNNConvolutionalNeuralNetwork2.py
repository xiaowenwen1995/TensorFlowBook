# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:19:47 2018

@author: xiaowenwen
"""

import cifar10, cifar10_input
import tensorflow as tf
import numpy as np
import time

#max_steps: the times of training
#batch_size: the size of batch
#data_dir: the directory of the dataset
max_steps = 3000
batch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'

#init weight
#shape: 
#stddev: standard deviation
#wl: weight loss control
#init use Normal distribution
#weight loss = wl * L2 loss(w)
#use a colloecion to save the weight loss
def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev = stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

#download the data set
cifar10.maybe_download_and_extract()

#create training dataset: Data Augmentation
images_train, labels_train = cifar10_input.distorted_inputs(data_dir = data_dir, batch_size = batch_size)

#create test dataset
images_test, labels_test = cifar10_input.inputs(eval_data = True, data_dir = data_dir, batch_size = batch_size)

#create placeholder to save image and label data
image_holder = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
label_holder = tf.placeholder(tf.int32, [batch_size])

#the first convolution layer
#weight1: init weight1: convolution kernel size:5x5, cannel:3 RGB, convlution kernel number: 64, standard deviation: 0.05, wl: 0
#kernel1: conv(images, W1), step is 1
#bias1: init 0
#conv1: activation function: ReLU(conv(images, W1) + b1)
#pool1: size: 3x3, step: 2x2
#norm1: LRN Local Response Normalization
weight1 = variable_with_weight_loss(shape = [5, 5, 3, 64], stddev = 5e-2, wl = 0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding = 'SAME')
bias1 = tf.Variable(tf.constant(0.0, shape = [64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')
norm1 = tf.nn.lrn(pool1, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)

#the second convolution layer: LRN before pool
#weight2: init weight2: convolution kernel size:5x5, cannel:64, convlution kernel number: 64, standard deviation: 0.05, wl: 0
#kernel2: conv(norm1, W2), step is 1
#bias2: init 0.1
#conv2: activation function: ReLU(conv(norm1, W2) + b2)
#norm2: LRN Local Response Normalization
#pool2: size: 3x3, step: 2x2
weight2 = variable_with_weight_loss(shape = [5, 5, 64, 64], stddev = 5e-2, wl = 0.0)
kernel2 = tf.nn.conv2d(norm1, weight2, [1, 1, 1, 1], padding = 'SAME')
bias2 = tf.Variable(tf.constant(0.1, shape = [64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
norm2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001 / 9.0, beta = 0.75)
pool2 = tf.nn.max_pool(norm2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

#fully connected layer
#reshape: flatten the vector to one dimensional
#dim: get the flattened length
#weight3: init weight3, standard deviation: 0.04, wl: 0.004
#bias3: init 0.1
#local3: activation function: ReLU(W3*reshape+b3)
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape = [dim, 384], stddev = 0.04, wl = 0.004)
bias3 = tf.Variable(tf.constant(0.1, shape = [384]))
local3 = tf.nn.relu(tf.matmul(reshape, weight3) + bias3)

#fully connected layer
#weight4: init weight4, standard deviation: 0.04, wl: 0.004
#bias4: init 0.1
#local4: activation function: ReLU(W4*local3+b4)
weight4 = variable_with_weight_loss(shape = [384, 192], stddev = 0.04, wl = 0.004)
bias4 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3, weight4) + bias4)

#classification layer
#weight5: init weight5, standard deviation: 1 / 192.0, wl: 0
#bias5: init 0
#logits: W5*local4+b5
weight5 = variable_with_weight_loss(shape = [192, 10], stddev = 1 / 192.0, wl = 0.0)
bias5 = tf.Variable(tf.constant(0.0, shape = [10]))
logits = tf.add(tf.matmul(local4, weight5), bias5)

#cross_entropy: softmax and cross entropy
#cross_entropy_mean: mean
#use a colloecion to save the total loss
def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels, name = 'cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'), name = 'total_loss')

#create loss
#train_op: Optimizer: Adam, learning rate 0.001
#top_k_op: calculate the top k
loss = loss(logits, label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits, label_holder, 1)

#create session
#run
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#start the queue for Data Augmentation
tf.train.start_queue_runners()

#training
for step in range(max_steps):
    start_time = time.time()
    image_batch, label_batch = sess.run([images_train, labels_train])
    _, loss_value = sess.run([train_op, loss], feed_dict = {image_holder: image_batch, label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        
        format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch')
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))

#testing        
num_examples = 10000
import math
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch, label_batch = sess.run([images_test, labels_test])
    predictions = sess.run([top_k_op], feed_dict = {image_holder: image_batch, label_holder: label_batch})
    true_count += np.sum(predictions)
    step += 1

#calculate the accuracy of the precision
precision = true_count / total_sample_count
print('precistion @ 1 = %.3f' % precision)
