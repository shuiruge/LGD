#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A raw implementation of the algorithm in the "READMD.md".

REMARK:
  * I do NOT think this can be implemented by inheriting `tf.train.Optimizer`,
    since it is a function (or say, `Op`-constructor), i.e. the function
    `make_loss_and_gradients`, that are passed into our optimizer, rather than
    simply a `Tensor` `loss`, as in `tf.train.Optimizer`.
"""

import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.rnn import ( GRUCell, LSTMCell,
                                     DropoutWrapper, MultiRNNCell )
from tensorflow.contrib.layers import fully_connected



mnist_data = input_data.read_data_sets(
    '../dat/mnist/', one_hot=True, source_url='../dat/mnist/')



# Parameter
INIT_N_SUB_TRAINS = 5
DECAY_STEPS = 300
DECAY_RATE = 0.5
MIN_N_SUB_TRAINS = 1
N_HISTS = 10
_EPSILON = -0.1
N_TRIALS = 10

n_inputs = 28*28
n_classes = 10
n_hiddens = 128

n_iters = 100000
batch_size = 128



class Optimizee(object):


    def __init__(self):

        with tf.name_scope('data'):
            self.inputs = tf.placeholder(shape=[None, 28*28], dtype='float32',
                                         name='inputs')
            self.targets = tf.placeholder(shape=[None, n_classes], dtype='int32',
                                          name='targets')

    def make_loss_and_gradients(self, w_h, w_a, b_h, b_a):
        """Test.

        Args:
            x: List of `Tensor`-like objects.

        Returns:
            `Op`s for loss and gradients.
        """

        with tf.name_scope('logits'):
            hidden = tf.nn.sigmoid(tf.matmul(self.inputs, w_h) + b_h)
            logits = tf.matmul(hidden, w_a) + b_a

        with tf.name_scope('loss'):
            regularization = sum(
                tf.reduce_mean(tf.square(_)) for _ in [w_h, w_a, b_h, b_a]
            )
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                       logits=logits, labels=self.targets))
            loss = loss + regularization

        with tf.name_scope('gradients'):
            var_list = [w_h, w_a, b_h, b_a]
            grad_list = tf.gradients(loss, var_list)
            gradients = [(_, grad_list[i]) for i, _ in enumerate(grad_list)]

        return loss, gradients





class Supervisor(object):

    def __init__(self):
        self.current_delta_loss = None

    def observe(self, delta_loss):
        self.current_delta_loss = delta_loss

    def keep_training(self):

        if self.current_delta_loss is None:
            return False

        else:
            if self.current_delta_loss > _EPSILON:
                return True
            else:
                return False



def test_StandardOptimizer(optimizee, optimizer, opt_name='standard',
                           n_iters=n_iters, bach_size=batch_size):

    with tf.name_scope('Variables'):
        w_h = tf.Variable(np.random.normal(size=[n_inputs, n_hiddens]),
                          dtype='float32', name='w_h')
        w_a = tf.Variable(np.random.normal(size=[n_hiddens, n_classes]),
                          dtype='float32', name='w_a')
        b_h = tf.Variable(np.random.normal(size=[n_hiddens]),
                          dtype='float32', name='b_h')
        b_a = tf.Variable(np.random.normal(size=[n_classes]),
                          dtype='float32', name='b_a')
    loss, gradients = optimizee.make_loss_and_gradients(w_h, w_a, b_h, b_a)
    train_op = optimizer.minimize(loss)

    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter('../dat/logdir/{}'.format(opt_name),
                                       sess.graph)

        previous_loss_val = float('inf')
        sess.run(init)
        time_start = time.time()

        for step in range(n_iters):

            # Get `feed_dict`
            X, y = mnist_data.train.next_batch(batch_size)
            feed_dict = {optimizee.inputs: X, optimizee.targets: y}

            loss_val, summary_val, _ = sess.run(
                [loss, summary_op, train_op], feed_dict=feed_dict)
            writer.add_summary(summary_val, step)

            delta_loss_val = loss_val - previous_loss_val
            previous_loss_val = loss_val

            print(step, loss_val, delta_loss_val)

            if loss_val < 0.1:
                break

    elapsed_time = time.time() - time_start
    print('Elapsed time: {} sec.'.format(elapsed_time))
    # => Elapsed time: XXX sec.




def test_LGDOptimizer(optimizee, n_iters=n_iters, batch_size=batch_size):

    trajectory = tf.placeholder('float32', shape=[1, N_HISTS, 3],
                                name='trajectory')
    args = [
        tf.placeholder('float32', shape=[n_inputs, n_hiddens], name='w_h'),
        tf.placeholder('float32', shape=[n_hiddens, n_classes], name='w_a'),
        tf.placeholder('float32', shape=[n_hiddens], name='b_h'),
        tf.placeholder('float32', shape=[n_classes], name='b_a'),
    ]

    with tf.name_scope('model'):

        def m(trajectory, rnn=GRUCell(5)):

            with tf.name_scope('rnn_layers'):
                # shape of `rnn_out`: `[1, N_HISTS, XXX]`
                rnn_out, state = tf.nn.dynamic_rnn(
                    cell=rnn, inputs=trajectory,
                    sequence_length=[N_HISTS], dtype='float32')
                # shape of `rnn_out`: `[1, XXX]`
                rnn_out = tf.unstack(rnn_out, axis=1)[-1]
                
            with tf.name_scope('hidden_layers'):
                # shape of `hidden`: `[1, XXX]`
                hidden = fully_connected(
                    rnn_out, 32, activation_fn=tf.nn.sigmoid)
            with tf.name_scope('output_layer'):
                # shape: `[1, 2]`
                z = fully_connected(hidden, 2, activation_fn=None)

            with tf.name_scope('momentum_and_lr'):
                # shapes: scalars
                momentum, learning_rate = tf.unstack(tf.squeeze(z))
            return momentum, learning_rate

    with tf.name_scope('meta_loss'):
        momentum, learning_rate = m(trajectory)
        meta_loss = 0.0
        iter_args = args
        for i in range(N_TRIALS):
            loss, gradients = optimizee.make_loss_and_gradients(*iter_args)
            meta_loss += loss
            iter_args = [
                x + momentum + learning_rate * g
                for x, g in gradients
            ]
        meta_loss = meta_loss / N_TRIALS

    # Update `w` with standard gradient descent optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(0.1)
        train_op = optimizer.minimize(meta_loss)

    with tf.name_scope('delta_loss'):
        delta_loss = meta_loss - loss

    # Initialization
    args_val = [
        np.random.normal(size=_.shape).astype('float32')
        for _ in args
    ]
    trajectory_val = np.zeros([1, N_HISTS, 3], dtype='float32')
    n_adjustments = 0

    tf.summary.scalar('loss', loss)
    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        writer = tf.summary.FileWriter('../dat/logdir/lgd', sess.graph)

        sess.run(init)
        time_start = time.time()
        X, y = mnist_data.train.next_batch(batch_size)

        for step in range(n_iters):

            # Get `feed_dict`
            data_feed_dict = {
                optimizee.inputs: X.astype('float32'),
                optimizee.targets: y.astype('int32'),
            }
            args_feed_dict = {
                arg: arg_val for arg, arg_val in list(zip(args, args_val))
            }
            trajectory_feed_dict = {
                trajectory: trajectory_val,
            }
            feed_dict = {
                **data_feed_dict,
                **args_feed_dict,
                **trajectory_feed_dict,
            }

            momentum_val, learning_rate_val, delta_loss_val, loss_val = \
                sess.run([momentum, learning_rate, delta_loss, meta_loss],
                         feed_dict=feed_dict)
            print(step, loss_val, delta_loss_val,
                  momentum_val, learning_rate_val, n_adjustments)

            summary_val, _ = sess.run(
                [summary_op, train_op], feed_dict=feed_dict)
            writer.add_summary(summary_val, step)

            if (step+1) % 30 != 0:
                # Keep adjusting
                n_adjustments += 1
                continue

            else:
                # Update `args`
                args_val = sess.run(iter_args, feed_dict=feed_dict)
                # Update `trajectory`
                new_trajectory_tail = np.asarray(
                    [[[momentum_val, learning_rate_val, delta_loss_val]]])
                trajectory_val = np.concatenate(
                    ( trajectory_val[:,1:,:], new_trajectory_tail ),
                    axis=1
                )

                X, y = mnist_data.train.next_batch(batch_size)

            if loss_val < 0.1:
                break

    elapsed_time = time.time() - time_start
    print('Elapsed time: {} sec.'.format(elapsed_time))
    print('Adjust {} times in total.'.format(n_adjustments))
    # => Elapsed time: 25.550095081329346 sec.




if __name__ == '__main__':

    tf.reset_default_graph()

    optimizee = Optimizee()
    test_LGDOptimizer(optimizee)
    #test_StandardOptimizer(optimizee, tf.train.RMSPropOptimizer(0.01),
    #                       opt_name='rmsprop')
