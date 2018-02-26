#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A raw implementation of the fine-tuner that I used to make."""

import abc
import numpy as np
import tensorflow as tf


class BaseSupervisor(abc.ABC):

  delta_loss_vals = []


  def observe(self, delta_loss_val):
    """
    Args:
      delta_loss_val: Float.
    """
    self.delta_loss_vals.append(delta_loss_val)


  @abc.abstractmethod
  def needs_fine_tuning(self):
    """Returns `True` if a fine-tuning is called for, else `False`."""
    pass


class Supervisor(BaseSupervisor):

  def __init__(self, n_smearing=5, tolerance=1e-2):

    self.n_smearing = n_smearing
    self.tolerance = tolerance


  def needs_fine_tuning(self):

    if len(self.delta_loss_vals) > 2 * self.n_smearing:

      current_delta_loss = np.mean(self.delta_loss_vals[-self.n_smearing:])
      mean_delta_loss = np.mean(self.delta_loss_vals)
      
      # Recall delta-loss shall be negative
      if current_delta_loss > mean_delta_loss * self.tolerance:
        return True

    else:
      return False


class BaseOptimizer(object):

  def __init__(self, make_loss_and_gradients, name='Optimizer'):
    """
    Args:
      make_loss_and_gradients:
        Callable that returns loss (as `Tensor`) and list of variable-gradient
        pairs (as tuple of two `Tensor`s).
      name:
        String.
    """
    self.make_loss_and_gradients = make_loss_and_gradients
    self.name = name


  @abc.abstractmethod
  def update(self, variables, parameters, name='update'):
    """
    Args:
      variables:
        List of `Tensor`s.
      parameters:
        List of `Tensor`s.

    Returns:
      List of `Tensor`s like the argument `variables`.

    Implement Example:
      ```
      with tf.name_scope(self.name):

        with tf.name_scope(name):

          loss, gradients = self.make_loss_and_gradients(variables)
          next_variables = [v - learning_rate * g for v, g in gradients]

      return next_variables
      ```
    """
    pass


class GradientDescentOptimizer(BaseOptimizer):

  def __init__(self, make_loss_and_gradients,
               name='GradientDescentOptimizer'):

    super().__init__(make_loss_and_gradients, name)


  def update(self, variables, parameters, name='update'):

    learning_rate, *rests = parameters

    with tf.name_scope(self.name):

      with tf.name_scope(name):

        loss, gradients = self.make_loss_and_gradients(*variables)
        next_variables = [v - learning_rate * g for v, g in gradients]

    return next_variables


class BaseFineTuner(abc.ABC):

  def __init__(self, n_trials, optimizer, make_loss_and_gradients,
               name='FineTuner'):
    """
    Args:
      n_trials:
        Interger.
      optimizer:
        An instance of a class inheriting the `BaseOptimizer`.
      make_loss_and_gradients:
        Callable that returns loss (as `Tensor`) and list of variable-gradient
        pairs (as tuple of two `Tensor`s).
      name:
        String.
    """

    self.n_trials = n_trials
    self.optimizer = optimizer
    self.make_loss_and_gradients = make_loss_and_gradients
    self.name = name


  @abc.abstractmethod
  def tune(self, parameters, name='Tune'):
    """
    Args:
      parameters:
        Tuple of `Tensor`s.
      name:
        String.

    Returns:
      Tuple of `Tensor`s, like the argument `parameters`, as the tuned.

    Implement Example:
      ```
      with tf.name_scope(self.name):

        with tf.name_scope(name):

          tuned_parameters = ...  # some implementation.

      return tuned_parameters
      ```
    """
    pass


  def make_meta_loss(self, parameters, current_varaiables, name='MetaLoss'):
    """
    Args:
      parameters:
        Tuple of `Tensor`s, with its length equals to `self.n_parameters`.
      current_varaiables:
        Tuple of `Tensor`s that can be feed to `self.make_loss_and_gradients`.
      name:
        String.

    Returns:
      A scalar.
    """
    with tf.name_scope(self.name):

      with tf.name_scope(name):

        tuned_parameters = self.tune(parameters)

        # Initialize
        variables = current_varaiables
        meta_loss = 0.0

        for trial in range(self.n_trials):
            next_variables = self.optimizer.update(variables, tuned_parameters)
            next_loss, next_gradients = \
                self.make_loss_and_gradients(*next_variables)
            meta_loss += next_loss
            variables = next_variables

    return meta_loss


class FineTuner(BaseFineTuner):

  def __init__(self, n_parameters, optimizer_for_tuner=None, **kwargs):
    """
    Args:
      n_parameters:
        Integer.
      kwargs:
        That of `super().__init__`.
    """
    super.__init__(**kwargs)

    with tf.name_scope(self.name):

      self.n_parameters = n_parameters

      if optimizer_for_tuner is None:
        self.optimizer_for_tuner = tf.train.AdamOptimizer(0.01)
      else:
        self.optimizer_for_tuner = optimizer_for_tuner


  def tune(self, parameters, name='Tune'):

    if len(parameters) != self.n_parameters:
      raise TypeError('Length of the argument `parameters` shall '
                      'be equal to the `self.n_parameters`.')

    with tf.name_scope(self.name):

      with tf.name_scope(name):

        with tf.name_scope('TunedParameters'):

          tuned_parameters = [
              tf.Variable(initial_value=p, name=p.name)
              for p in parameters
          ]

        with tf.name_scope('TuneOp'):
          
          meta_loss = self.make_meta_loss(parameters, current_varaiables)

          tune_op = self.optimizer_for_tuner.minimize(self.meta_loss)


    return tuned_parameters



def main():

  from tensorflow.examples.tutorials.mnist import input_data

  tf.reset_default_graph()

  mnist_data = input_data.read_data_sets(
      '../dat/mnist/', one_hot=True, source_url='../dat/mnist/')



  n_inputs = 28*28
  n_classes = 10
  n_hiddens = 128

  with tf.name_scope('data'):
    inputs = tf.placeholder(shape=[None, n_inputs], dtype='float32',
                            name='inputs')
    targets = tf.placeholder(shape=[None, n_classes], dtype='int32',
                             name='targets')

  def make_loss_and_gradients(w_h, w_a, b_h, b_a):
    """Test.

    Args:
      x: List of `Tensor`-like objects.

    Returns:
      `Op`s for loss and gradients.
    """

    with tf.name_scope('logits'):
      hidden = tf.nn.sigmoid(tf.matmul(inputs, w_h) + b_h)
      logits = tf.matmul(hidden, w_a) + b_a

    with tf.name_scope('loss'):
      regularization = sum(
          tf.reduce_mean(tf.square(_)) for _ in [w_h, w_a, b_h, b_a])
      loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits=logits, labels=targets))
      loss = loss + regularization

    with tf.name_scope('gradients'):
      var_list = [w_h, w_a, b_h, b_a]
      grad_list = tf.gradients(loss, var_list)
      gradients = [(_, grad_list[i]) for i, _ in enumerate(grad_list)]

    return loss, gradients

  variables = [
      tf.placeholder('float32', shape=[n_inputs, n_hiddens], name='w_h'),
      tf.placeholder('float32', shape=[n_hiddens, n_classes], name='w_a'),
      tf.placeholder('float32', shape=[n_hiddens], name='b_h'),
      tf.placeholder('float32', shape=[n_classes], name='b_a'),
  ]
  
  supervisor = Supervisor()
  
  optimizer = GradientDescentOptimizer(make_loss_and_gradients)
  learning_rate = tf.placeholder('float32', shape=[], name='learning_rate')
  parameters = [learning_rate]
  
  finetuner = FineTuner(n_trials=5, optimizer=optimizer, n_parameters=1,
                        make_loss_and_gradients=make_loss_and_gradients)
  meta_loss = finetuner.make_meta_loss(parameters, varaiables)


  tfopt = tf.train.AdamOptimizer(0.01)
  finetune_op = 
  init = tf.global_variables_initializer()

  with tf.Session() as sess:

    # Initialization
    sess.run(init)
    variable_values = [
        np.random.normal(size=v.shape).astype('float32')
        for v in variables
    ]

    for step in range(n_iters):
      
      X, y = mnist_data.train.next_batch(batch_size)

      # Get `feed_dict`
      data_feed_dict = {
          inputs: X.astype('float32'),
          targets: y.astype('int32'),
      }
      variables_feed_dict = {
          var: val for var, val in list(zip(variables, variable_values))
      }
      feed_dict = {
          **data_feed_dict,
          **variables_feed_dict,
      }




    

if __name__ == '__main__':

  main()