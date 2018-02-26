#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements the main Optimizer of learning gradient descent."""


import numpy as np
import tensorflow as tf
from collections import deque, namedtuple



class MetaOptimizer(object):
  """XXX"""
    
  def __init__(self, memory_size=100, name='MetaOptimizer'):
      
    with tf.name_scope(name):
        
      self.memory_size = memory_size
      
      
    
    
  def make_meta_loss():
     pass
        
    
        
        
        
        

        