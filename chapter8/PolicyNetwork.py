# -*- coding: utf-8 -*-
"""
Created on Tue May 29 16:49:02 2018

@author: xiaowenwen
"""

import numpy as np
import tensorflow as tf
import gym

#create environment: CartPole
env = gym.make('CartPole-v0')

#random action as baseline
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 2))
    reward_sum += reward
    #done == True means fail, break and reset
    if done:
        random_episodes += 1
        print("Reward for this episode was:", reward_sum)
        reward_sum = 0
        env.reset()
        
#H: hidden layer node: 50
#batch_size: the size of batch: 25
#learning_rate: learning rate: 0.1
#D: the Dimension of observation: 4
#gamma: discount of reward: 0.99
H = 50
batch_size = 25
learning_rate = 1e-1
D = 4
gamma = 0.99

#W1: the weight of hidden layer
#layer1: activation function: ReLu, ReLU(W1 * observations)
#W2: the weight of Sigmoid
#score: W2 * layer1
#probability: activation function: Sigmoid, Sigmoid(W2 * layer1)
observations = tf.placeholder(tf.float32, [None, D], name = "input_x")
W1 = tf.get_variable("W1", shape = [D, H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape = [H, 1], initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

#calculate the discount rewards of action
#running_add: the future discount rewards of action
#running_add * gamma + r[t]
def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

#input_y: the virtual label
#advantages: the potential value of this action
#loglik: 
#loss: loglik * advantages
input_y = tf.placeholder(tf.float32, [None, 1], name = "input_y")
advantages = tf.placeholder(tf.float32, name = "reward_signal")
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)

#tvars: trainable variables
tvars = tf.trainable_variables()
newGrads = tf.gradients(loss, tvars)

#optimizer: adam
#W1Grad:
#W2Grad:
#updateGrads:
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
W1Grad = tf.placeholder(tf.float32, name = "batch_grad1")
W2Grad = tf.placeholder(tf.float32, name = "batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

#xs: environment information
#ys: label list
#drs: action reward
#total_episodes: total number of episode
xs, ys, drs = [], [], []
reward_sum = 0
episode_number = 1
total_episodes = 10000

#create session and init, run
with tf.Session() as sess:
    rendering = False
    init = tf.global_variables_initializer()
    sess.run(init)
    observation = env.reset()
    
    #get trainable variables
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0
    
    #training loop    
    while episode_number <= total_episodes:
        #show
        if reward_sum / batch_size > 100 or rendering == True:
            env.render()
            rendering = True
        x = np.reshape(observation, [1, D])
        
        #get the probability of action = 1 and random
        tfprob = sess.run(probability, feed_dict = {observations: x})
        action = 1 if np.random.uniform() < tfprob else 0
        
        #create virtual label
        #y: virtual label: label = 1 - action
        xs.append(x)
        y = 1 - action
        ys.append(y)
        
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        
        #epx: observation
        #epy: label
        #epr: reward
        if done:
            episode_number += 1
            epx = np.vstack(xs)
            epy = np.vstack(ys)
            epr = np.vstack(drs)
            xs, ys, drs = [], [], []
            
            #discount_epr
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            
            #Gradient descent: epx, epy, epr
            tGrad = sess.run(newGrads, feed_dict = {observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad
            
            #update gradient
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict = {W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                
                print('Average reward for episode %d: %f.' % (episode_number, reward_sum / batch_size))
                
                #when reward > 200, finish this task
                if reward_sum / batch_size > 200:
                    print("Task solved in", episode_number, 'episodes!')
                    break
                
                reward_sum = 0
                
            observation = env.reset()