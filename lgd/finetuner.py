#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implements fine-tuner by gradient descent."""


import abc
import tensorflow as tf


class BaseOptimizee(abc.ABC):

  def __init__(self, name='BaseOptimizee'):
    self.name = name


  @abc.abstractmethod
  def make_loss_and_gradients(self, variables, name='LossAndGradients'):
    """If denote :math:`f` the loss-function and :math`\theta` the variables,
    this function shall returns :math:`f(\theta)` and :math:`\nabla f(\theta)`.

    Args:
      variables: list of tensors.

    Returns:
      A tuple of two elements. The first is a scalar (as the :math:`f`). And
      the later is a list of pairs of tensors; in the i-th pair, the first is
      the i-th element in the `variables`, say :math:`\theta_i` and the later
      is the :math:`\partial_{\theta_i} f (\theta)`.
    """
    with tf.name_scope(self.name):
      with tf.name_scope(name):
        pass



class FineTuer(object):

  def __init__(self, optimizee, optimizer, variables, parameters,
               name='FineTuner'):
    """Implements the fine-tuner by gradient descent.

    Args:
      optimizee: An instance of class inheriting the `BaseOptimizee`.
      optimizer: An instance of class inheriting the `BaseOptimizer`.
      variables: list of tensors as the argument of the method
          `optimizee.make_loss_and_gradients`.
      parameters: list of tensors as the argument of the method
          `optimizer.update`.
      name: string, optional.
    """

    self.name = name

    with tf.name_scope(name):
      self.variables = [
          tf.Variable(initial_value=_) for _ in variables
      ]
      self.parameters = [
          tf.Variables(initial_value=_) for _ in parameters
      ]


  def initialize(self, variables, parameters, name='Initialize'):

    with tf.name_scope(self.name):

      with tf.name_scope(name):

        # ...
        pass



  def make_meta_loss(self, n_trials=10, name='MetaLoss'):

    with tf.name_scope(self.name):

      with tf.name_scope(name):

        meta_loss = 0.0
        # ...


  def tune(self, name='TuneOp'):
