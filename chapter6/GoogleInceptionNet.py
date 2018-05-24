# -*- coding: utf-8 -*-
"""
Created on Tue May 22 09:27:32 2018

@author: xiaowenwen
"""

import tensorflow as tf
slim = tf.contrib.slim

#define a function to create a normal distribution
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

#set default parameters
#weight_decay: the decay of weight: 0.00004
#stddev: standard deviation: 0.1
#batch_norm_var_collection:
#batch_norm_params:
def inception_v3_arg_scope(weight_decay = 0.00004, stddev = 0.1, batch_norm_var_collection = 'moving_vars'):
    batch_norm_params = {
            'decay': 0.9997,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': {
                    'beta': None,
                    'gamma': None,
                    'moving_mean': [batch_norm_var_collection],
                    'moving_variance': [batch_norm_var_collection],
            }
    }
    
    #set the parameters of function automatically        
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer = slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weights_initializer = tf.truncated_normal_initializer(stddev = stddev), activation_fn = tf.nn.relu, normalizer_fn = slim.batch_norm, normalizer_params = batch_norm_params) as sc:
            return sc

#create the structure of Inception Net 
#inputs: input images tensor
def inception_v3_base(inputs, scope = None):
    #A dictionary table to save some key nodes
    end_points = {}
    
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        #set the parameters of conv2d, max_pool2d, avg_pool2d automatically
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'VALID'):
            #299x299x3 => 149x149x32
            net = slim.conv2d(inputs, 32, [3, 3], stride = 2, scope = 'Conv2d_1a_3x3')
            #149x149x3 => 147x147x32
            net = slim.conv2d(net, 32, [3, 3], scope = 'Conv2d_2a_3x3')
            #147x147x32 => 147x147x64
            net = slim.conv2d(net, 64, [3, 3], padding = 'SAME', scope = 'Conv2d_2b_3x3')
            #147x147x64 => 73x73x64
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_3a_3x3')
            #73x73x64 => 73x73x80
            net = slim.conv2d(net, 80, [1, 1], scope = 'Conv2d_3b_1x1')
            #73x73x80 => 71x71x192
            net = slim.conv2d(net, 192, [3, 3], scope = 'Conv2d_4a_3x3')
            #71x71x192 => 35x35x192
            net = slim.max_pool2d(net, [3, 3], stride = 2, scope = 'MaxPool_5a_3x3')
        
        #Inception Module Group
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
            #output: 35x35x256
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #output: 35x35x288
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #output: 35x35x288
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 48, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 64, [5, 5], scope = 'Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope = 'Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #output: 17x17x768
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 384, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 64, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'MaxPool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            
            #output: 17x17x768
            with tf.variable_scope('Mixed_6b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 128, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 128, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 128, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 128, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #output: 17x17x768
            with tf.variable_scope('Mixed_6c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0a_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #output: 17x17x768
            with tf.variable_scope('Mixed_6d'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 160, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192,[7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 160, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 160, [1, 7], scope = 'Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 160, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            
            #output: 17x17x768
            with tf.variable_scope('Mixed_6e'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope = 'Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0c_1X7')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope = 'Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 192, [1, 7], scope = 'Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            end_points['Mixed_6e'] = net
            
            #output: 8x8x1280
            with tf.variable_scope('Mixed_7a'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 320, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 192, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 192, [1, 7], scope = 'Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 192, [7, 1], scope = 'Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 192, [3, 3], stride = 2, padding = 'VALID', scope = 'Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride = 2, padding = 'VALID', scope = 'Maxpool_1a_3x3')
                net = tf.concat([branch_0, branch_1, branch_2], 3)
            
            #output: 8x8x2048
            with tf.variable_scope('Mixed_7b'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0b_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_3], 3)
            
            #output: 8x8x2048
            with tf.variable_scope('Mixed_7c'):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, 320, [1, 1], scope = 'Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, 384, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_1 = tf.concat([slim.conv2d(branch_1, 384, [1, 3], scope = 'Conv2d_0b_1x3'), slim.conv2d(branch_1, 384, [3, 1], scope = 'Conv2d_0c_3x1')], 3)
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, 448, [1, 1], scope = 'Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope = 'Conv2d_0b_3x3')
                    branch_2 = tf.concat([slim.conv2d(branch_2, 384, [1, 3], scope = 'Conv2d_0c_1x3'), slim.conv2d(branch_2, 384, [3, 1], scope = 'Conv2d_0d_3x1')], 3)
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope = 'AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope = 'Conv2d_0b_1x1')
                net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
            return net, end_points

#the structure of Inception Net
#inputs: input dataset tensor
#num_classes: the number of classification
#dropout_keep_prob: dropout keep probability: 0.8
#prediction_fn: classfication function: softmax
#spatial_squeeze: remove dimensions with a dimension of one
#reuse: no
#scope: 
def inception_v3(inputs, num_classes = 1000, is_training = True, dropout_keep_prob = 0.8, prediction_fn = slim.softmax, spatial_squeeze = True, reuse = None, scope = 'InceptionV3'):
    with tf.variable_scope(scope, 'InceptionV3', [inputs, num_classes], reuse = reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training = is_training):
            net, end_points = inception_v3_base(inputs, scope = scope)
            
            #Auxiliary Logits
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding = 'SAME'):
                aux_logits = end_points['Mixed_6e']
                with tf.variable_scope('AuxLogits'):
                    #17x17x768 => 5x5x768
                    aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride = 3, padding = 'VALID', scope = 'AvgPool_1a_5x5')
                    aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope = 'Conv2d_1b_1x1')
                    
                    #5x5x768 => 1x1x768
                    aux_logits = slim.conv2d(aux_logits, 768, [5, 5], weights_initializer = trunc_normal(0.01), padding = 'VALID', scope = 'Conv2d_2a_5x5')
                    #1x1x768 => 1x1x1000
                    aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, weights_initializer = trunc_normal(0.001), scope = 'Conv2d_2b_1x1')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name = 'SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits
            
            #average pool and softmax
            with tf.variable_scope('Logits'):
                #=> 1x1x2048
                net = slim.avg_pool2d(net, [8, 8], padding = 'VALID', scope = 'AvgPool_1a_8x8')
                net = slim.dropout(net, keep_prob = dropout_keep_prob, scope = 'Dropout_1b')
                end_points['PreLogits'] = net
                #=> 1x1x1000
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'Conv2d_1c_1x1')
                #1x1x1000 =>1000
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name = 'SpatialSqueeze')
            end_points['Logits'] = logits
            #prediction: softmax
            end_points['Predictions'] = prediction_fn(logits, scope = 'Predictions')
    return logits, end_points


#run
from datetime import datetime
import math
import time

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

batch_size = 32
height, width = 299, 299
inputs = tf.random_uniform((batch_size, height, width, 3))
with slim.arg_scope(inception_v3_arg_scope()):
    logits, end_points = inception_v3(inputs, is_training = False)

#init
init = tf.global_variables_initializer()
#create session and save it
sess = tf.Session()
sess.run(init)
num_batches = 100
#run
time_tensorflow_run(sess, logits, "Forward")                
                
                