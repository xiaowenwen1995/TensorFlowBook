#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:15:47 2018

@author: xiaowenwen
"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#init: xavier initialization 
#xavier initialization: Gaussian or Uniform distribution and weight = 0 and variance(方差) = 2 / (n_in + n_out)
#here: Uniform Distribution between (-sqrt(6 / (n_in + n_out)), sqrt(6 / (n_in + n_out)))
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    
    #input parameters
    #n_input: input lay number
    #n_hidden: hidden lay number
    #transfer_function: activation function, default:softplus
    #optimizer: optimization function, default: adam
    #scale: Gaussian noise figure, default:0.1
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        
        #init weights
        network_weights = self._initialize_weights()
        self.weights = network_weights
        
        #add noise, active and reconstruction
        #add noise: [x+scale *random(n_input)]W1+b1
        #reconstruction: W2*hidden+b2
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        
        #loss function: squared error
        #optimization function
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        #init and create session, run
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    #init weights
    #w1 use xavier initialization and others are zero
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
    
    #run loss and optimizer
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost
    
    #run loss to calculate loss
    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    #run hidden and return the result of hidden layer
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    #run reconstruction and recover data
    #input data: hidden
    def generate(self, hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    
    #run reconstruction 
    #this function include transform and generate
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})
    
    #get w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    
    #get b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])
    
#download dataset from mnist
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

#standardize the data
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

#get random block from data
def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

#standardize the train data and test data
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

#the number of samples
#max traning epoch: 20
#batch size: 128
#display step: 0, display the cost for each step 
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

#create an instance
#n_input 784
#n_hidden 200
#transfer function: softplus
#optimizer: adam and learning rate 0.001
#scale: 0.01
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = 200, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01)

#training
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
        
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

#print the lost of test data
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
