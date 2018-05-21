# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:02:14 2018

@author: xiaowenwen
"""

from datetime import datetime
import math
import time
import tensorflow as tf

#convolution layer
#input_op: input tensor
#name: the name of this layer
#kh: kernel height
#kw: kernel width
#n_out: convolution kernel number
#dh: step height
#dw: step width
#p: parameter
def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    #get the cannel number
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        #kernel: [kernel height, kernel width, cannel, kernel number]
        #init: xavier
        kernel = tf.get_variable(scope + "w", shape = [kh, kw, n_in, n_out], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer_conv2d())
        
        #conv: conv(input, kernel)
        #bias: init 0.0
        #z: conv(input, kernel) + b
        #activation: activation function: ReLU, ReLU(conv(input, kernel) + b)
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding = 'SAME')
        bias_init_val =  tf.constant(0.0, shape = [n_out], dtype = tf.float32)
        biases = tf.Variable(bias_init_val, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        p += [kernel, biases]
        return activation

#full connect layer
#input_op: input tensor
#name: the name of this layer
#n_out: output cannel
#p: parameter        
def fc_op(input_op, name, n_out, p):
    #get the cannel number of input data
    n_in = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        #kernel: [input cannel, output cannel]
        #bias: init 0.1
        #activation: activation function: ReLU, ReLU(input, kernel) + b
        kernel = tf.get_variable(scope + "w", shape = [n_in, n_out], dtype = tf.float32, initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape = [n_out], dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)
        p += [kernel, biases]
        return activation

#max pool function
#input_op: input tensor
#name: the name of this layer
#kh: kernel height
#kw: kernel width
#dh: step height
#dw: step width 
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op, ksize = [1, kh, kw, 1], strides = [1, dh, dw, 1], padding = 'SAME', name = name)

#the structure of VGGNet16
#input_op: input tensor
#keep_prob: keep probability for dropout 
def inference_op(input_op, keep_prob):
    p = []
    
    #layer1
    #convolution layer1: input_op: 224x224x3, kernel size: 3x3, cannel: 3, kernel number: 64, step: 1x1
    #convolution layer2: input_op: 224x224x64, kernel size: 3X3, cannel:64, kernel number: 64, step: 1x1
    #pool layer: input_op: 224x224x64, kernel size: 2x2, step: 2x2
    #output: 112x112x64
    conv1_1 = conv_op(input_op, name = "conv1_1", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    conv1_2 = conv_op(conv1_1, name = "conv1_2", kh = 3, kw = 3, n_out = 64, dh = 1, dw = 1, p = p)
    pool1 = mpool_op(conv1_2, name = "pool1", kh = 2, kw = 2, dw = 2, dh = 2)
    
    #layer2
    #convolution layer1: input_op: 112x112x64, kernel size: 3x3, cannel: 64, kernel number: 128, step: 1x1
    #convolution layer2: input_op: 112x112x128, kernel size: 3X3, cannel:128, kernel number: 128, step: 1x1
    #pool layer: input_op: 112x112x128, kernel size: 2x2, step: 2x2
    #output: 56x56x128
    conv2_1 = conv_op(pool1, name = "conv2_1", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    conv2_2 = conv_op(conv2_1, name = "conv2_2", kh = 3, kw = 3, n_out = 128, dh = 1, dw = 1, p = p)
    pool2 = mpool_op(conv2_2, name = "pool2", kh = 2, kw = 2, dw = 2, dh = 2)
    
    #layer3
    #convolution layer1: input_op: 56x56x128, kernel size: 3x3, cannel: 128, kernel number: 256, step: 1x1
    #convolution layer2: input_op: 56x56x256, kernel size: 3X3, cannel:256, kernel number: 256, step: 1x1
    #pool layer: input_op: 56x56x256, kernel size: 2x2, step: 2x2
    #output: 28x28x256
    conv3_1 = conv_op(pool2, name = "conv3_1", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    conv3_2 = conv_op(conv3_1, name = "conv3_2", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    conv3_3 = conv_op(conv3_2, name = "conv3_3", kh = 3, kw = 3, n_out = 256, dh = 1, dw = 1, p = p)
    pool3 = mpool_op(conv3_3, name = "pool3", kh = 2, kw = 2, dw = 2, dh = 2)
    
    #layer4
    #convolution layer1: input_op: 28x28x256, kernel size: 3x3, cannel: 256, kernel number: 512, step: 1x1
    #convolution layer2: input_op: 28x28x512, kernel size: 3X3, cannel:512, kernel number: 512, step: 1x1
    #pool layer: input_op: 28x28x512, kernel size: 2x2, step: 2x2
    #output: 14x14x512
    conv4_1 = conv_op(pool3, name = "conv4_1", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv4_2 = conv_op(conv4_1, name = "conv4_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv4_3 = conv_op(conv4_2, name = "conv4_3", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    pool4 = mpool_op(conv4_3, name = "pool4", kh = 2, kw = 2, dw = 2, dh = 2)
    
    #layer5
    #convolution layer1: input_op: 14x14x512, kernel size: 3x3, cannel: 256, kernel number: 512, step: 1x1
    #convolution layer2: input_op: 14x14x512, kernel size: 3X3, cannel:512, kernel number: 512, step: 1x1
    #pool layer: input_op: 14x14x512, kernel size: 2x2, step: 2x2
    #output: 7x7x512
    conv5_1 = conv_op(pool4, name = "conv5_1", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv5_2 = conv_op(conv5_1, name = "conv5_2", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    conv5_3 = conv_op(conv5_2, name = "conv5_3", kh = 3, kw = 3, n_out = 512, dh = 1, dw = 1, p = p)
    pool5 = mpool_op(conv5_3, name = "pool5", kh = 2, kw = 2, dw = 2, dh = 2)
    
    #reshape the vector to one dimensional 7x7x512 = 25088
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name = "resh1")
    
    #full connect layer1 and dropout
    #node number 4096
    #activation function: ReLU
    #dropout: 
    fc6 = fc_op(resh1, name = "fc6", n_out = 4096, p = p)
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name = "fc6_drop")
    
    #full connect layer2 and dropout
    #node number 4096
    #activation function: ReLU
    #dropout: 
    fc7 = fc_op(fc6_drop, name = "fc7", n_out = 4096, p = p)
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name = "fc7_drop")
    
    #full connect layer3 and softmax
    #node number: 1000
    #classification: softmax
    #argmax: find the category with the highest probability
    fc8 = fc_op(fc7_drop, name = "fc8", n_out = 1000, p = p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p

#run
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        session.run(target, feed_dict = feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' %(datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %(datetime.now(), info_string, num_batches, mn, sd))

#main function    
def run_benchmark():
    with tf.Graph().as_default():
        #create the random images 224x224
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype = tf.float32, stddev = 1e-1))
        
        #VGGNet
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)
        
        #init and save the session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        
        #run
        #loss function: l2 loss
        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")
    
batch_size = 32
num_batches = 100
#run the main function
run_benchmark()        