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


class FineTuner(object):

  def __init__(self, optimizee, optimizer, variables, parameters,
               opt_for_tuner=None, name='FineTuner'):
    """Implements the fine-tuner by gradient descent.

    Args:
      optimizee: An instance of class inheriting the `BaseOptimizee`.
      optimizer: A class inheriting the `tf.train.Optimizer`.
      variables: list of tensors as the argument of the method
          `optimizee.make_loss_and_gradients`.
      parameters: list of tensors as the argument of the method
          `optimizer.update`.
      opt_for_tuner: `None` or an instance of class inheriting the
          `tf.train.Optimizer`, optional.
      name: string, optional.
    """

    self.name = name

    with tf.name_scope(name):
      self.variables = [
          tf.Variable(initial_value=_) for _ in variables
      ]
      self.parameters = [
          tf.Variable(initial_value=_) for _ in parameters
      ]
      self.loss, self.gradients = optimizee\
          .make_loss_and_gradients(*self.variables)
      self.optimizer = optimizer(*self.parameters)
      # XXX: notice that args in `tf.train.Optimizer.__init__()` will be
      # converted to tensor by `tf.convert_to_tensor()` in its `_prepare()`.
      # This may cause error, to be revealed.

      if opt_for_tuner is None:
        self.opt_for_tuner = tf.train.AdamOptimizer(0.01)
      else:
        self.opt_for_tuner = opt_for_tuner

      # Initialize
      self.meta_loss = None


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


  def make_meta_loss(self, n_trials, name='MetaLoss'):

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


  def tune(self, n_trials=10, name='TuneOp'):

    with tf.name_scope(self.name):

      self.meta_loss = self.make_meta_loss(n_trials)

      with tf.name_scope(name):

        tune_op = self.opt_for_tuner.minimize(
            self.meta_loss, var_list=self.parameters)

    return tune_op



if __name__ == '__main__':


  class TestOptimizee(BaseOptimizee):

    def make_loss_and_gradients(self, x):
      loss = tf.square(x)
      g = tf.gradients(loss, [x])
      gradients = [(g, x)]
      return loss, gradients

  def test():
    optimizee = TestOptimizee()
    optimizer = tf.train.GradientDescentOptimizer
    x = tf.Variable(initial_value=1000.0, dtype='float32')
    lr = tf.Variable(initial_value=0.01, dtype='float32')
    finetuner = FineTuner(optimizee=optimizee, optimizer=optimizer,
                          variables=[x], parameters=[lr])

    tune_op = finetuner.tune()

    with tf.Session() as sess:

      for i in range(10):

        _, lr_val = sess.run([tune_op, lr])

        print(lr_val)


  test()





      
