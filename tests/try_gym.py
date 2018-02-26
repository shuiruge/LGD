#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Try to implement on `gym`"""


import gym
import abc
from collections import namedtuple, deque
import numpy as np
import tensorflow as tf


Point = namedtuple('Point', 'momentum learning_rate effect')


class BaseLGDOptimizer(abc.ABC):
    
    def __init__(self, memory_size, optimizer=None, dtype='float32'):
        
        self.memory_size = memory_size
        self.dtype = dtype
        
        self.memory = deque(maxlen=self.memory_size)
        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer()
        else:
            self.optimizer = optimizer
           
            
    @abc.abstractmethod
    def get_momentum_and_learning_rate(self):
        """Returns a tuple of two floats."""
        pass
        
    
    def train(self, n_iters):
        """Train the optimizer by gradient descent.
        
        Args:
            trajectory:
                List of tuples of `State`s.
            n_iters:
                Number of iterations in training.
        """
        pass
    
    
    
class Supervisor(abc.ABC):
    
    @abc.abstractmethod
    def get_n_iters(self):
        
        pass
            


    


class Agent(object):
    
    def __init__(self, optimizer, supervisor, memory_size=100):
        
        self.optimizer = optimizer
        self.supervisor = supervisor
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))