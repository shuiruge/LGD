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
      self.loss, self.gradients = optimizee\
          .make_loss_and_gradients(self.variables)
      self.optimizer = optimizer(*parameters)
      # XXX: notice that args in `tf.train.Optimizer.__init__()` will be
      # converted to tensor by `tf.convert_to_tensor()` in its `_prepare()`.
      # This may cause error, to be revealed.


  def reset(self, variables, parameters, name='Reset'):
    """XXX"""

    reset_ops = []

    with tf.name_scope(self.name):

      with tf.name_scope(name):

        for i, var in enumerate(self.variables):
          reset_ops.append(var.assign(variables[i]))

        for i, param in enumerate(self.parameters):
          reset_ops.append(param.assign(parameters[i]))

    return tf.group(reset_ops)


  def make_meta_loss(self, n_trials=10, name='MetaLoss'):

    with tf.name_scope(self.name):

      with tf.name_scope(name):

        # `tf.train.Optimizer.apply_gradients()` only updates the
        # `tf.Variable`s within its argument `grad_and_vars`
        update_op = self.optimizer.apply_gradients(self.gradients)

        def cond(trial, meta_loss):
          return tf.less(trial, n_trials)

        def body(trial, meta_loss):
          # Ensure that `update_op` is called before adding `loss`
          # to `meta_loss`
          with tf.control_dependencies([update_op]):
            return trial + 1, meta_loss + loss

        _, meta_loss = tf.while_loop(cond, body, [0, 0.0])

    return meta_loss


  def tune(self, name='TuneOp'):

    with tf.name_scope(self.name):

      with tf.name_scope(name)

      tune_op = self.opt_for_tuner.minimize(
          meta_loss, var_list=self.parameters)

    return tune_op
