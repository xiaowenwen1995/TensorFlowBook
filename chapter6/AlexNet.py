# -*- coding: utf-8 -*-
"""
Created on Fri May 18 11:20:26 2018

@author: xiaowenwen
"""
#this code only calculate the time-consuming

from datetime import datetime
import math
import time
import tensorflow as tf

#batch size: 32
#batch number: 100
batch_size = 32
num_batches = 100

#print the structure of each layer: the name and the size of tensor
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

#define the structure of AlexNet   
def inference(images):
    parameters = []
    
    #the first convolution layer
    #kernel: normal distribution: standard deviation: 0.1, convolution kernel size: 11x11, cannel: 3, convolution kernel number: 64
    #conv: conv(images, kernel), step: 4x4
    #biases: init 0
    #bias: conv(images, kernel) + biases
    #conv1: activation function: ReLU, ReLU(conv(images, kernel) + biases)
    #print the first convlution layer
    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [64], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name = scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    
    #lrn1: lrn(conv1)
    #pool1: max pool: pool size 3x3
    #print the first max pool
    lrn1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'lrn1')
    pool1 = tf.nn.max_pool(lrn1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool1')
    print_activations(pool1)
    
    #the second convolution layer
    #kernel: normal distribution: standard deviation: 0.1, convolution kernel size: 5x5, cannel: 64, convolution kernel number: 192
    #conv: conv(pool1, kernel), step: 1x1
    #biases: init 0
    #bias: conv(pool1, kernel) + biases
    #conv2: activation function: ReLU, ReLU(conv(pool1, kernel) + biases)
    #print the second convlution layer
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [192], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv2)
    
    #lrn2: lrn(conv2)
    #pool2: max pool: pool size 3x3
    #print the second max pool    
    lrn2 = tf.nn.lrn(conv2, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'lrn2')
    pool2 = tf.nn.max_pool(lrn2, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool2')
    print_activations(pool2)
    
    #the third convolution layer
    #kernel: normal distribution: standard deviation: 0.1, convolution kernel size: 3x3, cannel: 192, convolution kernel number: 384
    #conv: conv(pool2, kernel), step: 1x1
    #biases: init 0
    #bias: conv(pool2, kernel) + biases
    #conv3: activation function: ReLU, ReLU(conv(pool2, kernel) + biases)
    #print the third convlution layer
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [384], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv3)
    
    #the fourth convolution layer
    #kernel: normal distribution: standard deviation: 0.1, convolution kernel size: 3x3, cannel: 384, convolution kernel number: 256
    #conv: conv(conv3, kernel), step: 1x1
    #biases: init 0
    #bias: conv(conv3, kernel) + biases
    #conv4: activation function: ReLU, ReLU(conv(conv3, kernel) + biases)
    #print the fourth convlution layer
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv4)
    
    #the fifth convolution layer
    #kernel: normal distribution: standard deviation: 0.1, convolution kernel size: 3x3, cannel: 256, convolution kernel number: 256
    #conv: conv(conv4, kernel), step: 1x1
    #biases: init 0
    #bias: conv(conv4, kernel) + biases
    #conv5: activation function: ReLU, ReLU(conv(conv4, kernel) + biases)
    #print the fifth convlution layer   
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype = tf.float32, stddev = 1e-1), name = 'weights')
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding = 'SAME')
        biases = tf.Variable(tf.constant(0.0, shape = [256], dtype = tf.float32), trainable = True, name = 'biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name = scope)
        parameters += [kernel, biases]
        print_activations(conv5)
        
    #pool5: max pool: pool size 3x3
    #print the fifth max pool
    pool5 = tf.nn.max_pool(conv5, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'VALID', name = 'pool5')
    print_activations(pool5)
    
    return pool5, parameters

#calculate the time of each round
def time_tensorflow_run(session, target, info_string):
    #warm up round, Avoid initial video memory loading
    #total duration
    #the square of total duration
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    
    #the number of calculations is (num_batches + num_steps_burn_in)
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    
    #mn: average of time-consuming
    #sd: standard deviation of time-consuming
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mn, sd))

#main function
#dataset use default graph    
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype = tf.float32, stddev = 1e-1))
        
        #define the structure of AlexNet 
        pool5, parameters = inference(images)
        
        #init and create session, run
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        #calculate the time of each round
        time_tensorflow_run(sess, pool5, "Forward")
        
        #loss funtion: l2
        #calculate the gradient
        #calculate the time of each round
        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess, grad, "Forward-backward")

#run the main function
run_benchmark()
    

    