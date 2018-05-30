# -*- coding: utf-8 -*-
"""
Created on Wed May 30 10:56:08 2018

@author: xiaowenwen
"""

#create GridWorld
import numpy as np
import random
import itertools
import scipy.misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os
#%matplotlib inline

#object in environment
#coordinates: x, y
class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name

#environment        
class gameEnv():
    #size: the size of environment
    def __init__(self, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        a = self.reset()
        #show the observation
        plt.imshow(a, interpolation = "nearest")
    
    #create the role
    #hero: 1, goal(reward +1): 4, fire(reward -1): 2    
    def reset(self):
        self.objects = []
        hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
        self.objects.append(hero)
        goal = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal)
        hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole)
        goal2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal2)
        hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
        self.objects.append(hole2)
        goal3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal3)
        goal4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
        self.objects.append(goal4)
        state = self.renderEnv()
        self.state = state
        return state
    
    #move the hero
    #0=>up, 1=>down, 2=>left, 3=>right
    def moveChar(self, direction):
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        self.objects[0] = hero
    
    #choose a non-conflicting location
    #all position - current position, then random    
    def newPosition(self):
        iterables = [range(self.sizeX), range(self.sizeY)]
        points = []
        for t in itertools.product(* iterables):
            points.append(t)
        currentPositions = []
        for objectA in self.objects:
            if (objectA.x, objectA.y) not in currentPositions:
                currentPositions.append((objectA.x, objectA.y))
        for pos in currentPositions:
            points.remove(pos)
        location = np.random.choice(range(len(points)), replace = False)
        return points[location]
    
    #check if the hero touched the goal or fire
    #return reward and new a this goal or fire
    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'fire'))
                return other.reward, False
        return 0.0, False
    
    #draw a environment
    #create a picture: size: size +2, cannel: 3, init: 1(white)
    #add roles
    #resize: => 84x84x3
    def renderEnv(self):
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])
        a[1:-1, 1: -1, :] = 0
        hero = None
        for item in self.objects:
            a[item.y + 1: item.y + item.size + 1, item.x + 1: item.x + item.size + 1, item.channel] = item.intensity
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp = 'nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp = 'nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp = 'nearest')
        a = np.stack([b, c, d], axis = 2)
        return a
    
    #action
    def step(self, action):
        self.moveChar(action)
        reward, done = self.checkGoal()
        state = self.renderEnv()
        return state, reward, done

#create a environment    
env = gameEnv(size = 5)