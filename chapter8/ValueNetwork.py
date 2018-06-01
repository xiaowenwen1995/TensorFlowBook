# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:10:43 2018

@author: xiaowenwen
"""

import numpy as np
import random
import tensorflow as tf
import os

from GridWorld import gameEnv
env = gameEnv(size = 5)

class Qnetwork():
    def __init__(self, h_size):
        #scalarInput: 84x84x3 = 21168 => [-1, 84, 84, 3]
        self.scalarInput = tf.placeholder(shape = [None, 21168], dtype = tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape = [-1, 84, 84, 3])
        #convolution layer1: kernel size: 8x8, step: 4x4, cannel: 32, =>20x20x32
        #convolution layer2: kernel size: 4x4, step: 2x2, cannel: 64, =>9x9x64
        #convolution layer3: kernel size: 3x3, step: 1x1, cannel: 64, =>7x7x64
        #convolution layer4: kernel size: 7x7, step: 1x1, cannel: 512, => 1x1x512
        self.conv1 = tf.contrib.layers.convolution2d(inputs = self.imageIn, num_outputs = 32, kernel_size = [8, 8], stride = [4, 4], padding = 'VALID', biases_initializer = None)
        self.conv2 = tf.contrib.layers.convolution2d(inputs = self.conv1, num_outputs = 64, kernel_size = [4, 4], stride = [2, 2], padding = 'VALID', biases_initializer = None)
        self.conv3 = tf.contrib.layers.convolution2d(inputs = self.conv2, num_outputs = 64, kernel_size = [3, 3], stride = [1, 1], padding = 'VALID', biases_initializer = None)
        self.conv4 = tf.contrib.layers.convolution2d(inputs = self.conv3, num_outputs = 512, kernel_size = [7, 7], stride = [1, 1], padding = 'VALID', biases_initializer = None)
        
        #streamAC: Advantage Function(value which is caused by action)
        #streamVC: Value function(value which is caused by environment)
        #AW: weight of full connect layer
        #VW: weight of full connect layer
        #advantage: AW * streamA
        #value: VW * streamV
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = tf.contrib.layers.flatten(self.streamAC)
        self.streamV = tf.contrib.layers.flatten(self.streamVC)
        self.AW = tf.Variable(tf.random_normal([h_size // 2, env.actions]))
        self.VW = tf.Variable(tf.random_normal([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        
        #Qout = value + advantage - reduce_mean(advantage)
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, reduction_indices = 1, keep_dims = True))
        self.predict = tf.argmax(self.Qout, 1)
        
        #actions onehot encode
        #Q = Qout * actions_onehot
        self.targetQ = tf.placeholder(shape = [None], dtype = tf.float32)
        self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype = tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), reduction_indices = 1)
        
        #loss: reduce_mean(square(target - Q))
        #optimizer: Adam
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

#Experience Replay    
class experience_buffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
        
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

#84x84x3 => 21168x1    
def processState(states):
    return np.reshape(states, [21168])

#update DQN model parameters
def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0: total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

#batch_size: the size of batch
#update_freq: the update frequency
#y: discount factor on the target Q-values
#startE: random start Acion probability
#endE: random end Action probability
#anneling_steps: steps of reduce startE to endE
#num_episodes: the training steps
#pre_train_steps: random action test
#max_epLength: the number of action steps in each episode
#load_model: if read the training model before
#path: model path
#h_size: the full connect layer node number
#tau: target DQN learning rate
batch_size = 32
update_freq = 4
y = .99
startE = 1
endE = 0.1
anneling_steps = 10000.
num_episodes = 10000
pre_train_steps = 10000
max_epLength = 50
load_model = False
path = "./dqn"
h_size = 512
tau = 0.001 

#mainQN
#targetQN
#init
tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)
init = tf.global_variables_initializer()

trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

e = startE
stepDrop = (startE - endE) / anneling_steps

rList = []
total_steps = 0

#model saver
saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)

#create session and training
with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    sess.run(init)
    updateTarget(targetOps, sess)
    for i in range(num_episodes + 1):
        episodeBuffer = experience_buffer()
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        
        while j < max_epLength:
            j += 1
            #random action training
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0, 4)
            else:
                a = sess.run(mainQN.predict, feed_dict = {mainQN.scalarInput:[s]})[0]
            s1, r, d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            
            #DQN training
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)
                    A = sess.run(mainQN.predict, feed_dict = {mainQN.scalarInput:np.vstack(trainBatch[:, 3])})
                    Q = sess.run(targetQN.Qout, feed_dict = {targetQN.scalarInput:np.vstack(trainBatch[:, 3])})
                    doubleQ = Q[range(batch_size), A]
                    targetQ = trainBatch[:, 2] + y * doubleQ
                    _ = sess.run(mainQN.updateModel, feed_dict = {mainQN.scalarInput: np.vstack(trainBatch[:, 0]), mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})
                    updateTarget(targetOps, sess)
                    
            rAll += r
            s = s1
            
            if d == True:
                break
            
        myBuffer.add(episodeBuffer.buffer)
        rList.append(rAll)
        if i > 0 and i % 25 == 0:
            print('episode', i, ', average reward of last 25 episode', np.mean(rList[-25:]))
        if i > 0 and i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Saved Model")
    saver.save(sess, path + '/model-' + str(i) + '.cptk')     

#visual print
import matplotlib.pyplot as plt
    
rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)