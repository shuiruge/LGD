#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A raw implementation of the algorithm in the "READMD.md"."""

import numpy as np
import tensorflow as tf
import time


tf.reset_default_graph()


# Parameter
N_SUB_TRAINS = 3
N_HISTS = 30

n_dims = 10
n_iters = 100000


def make_f(x, n_dims=n_dims):
    """Test."""
    enlarge = tf.constant(
        np.array([10.0 if i < n_dims/2 else 1.0 for i in range(n_dims)]),
        dtype='float32')
    return tf.reduce_sum(tf.square(enlarge * x))


def main(make_f=make_f, n_dims=n_dims, n_iters=n_iters):
    
    env = tf.placeholder('float32', shape=[1, N_HISTS, 3])  # "environment".
    x = tf.placeholder('float32', shape=[1, n_dims])
    
    
    # Model
    # Use deep RNN, since shallow RNN cannot understand,
    # and thus sometimes makes `NaN`.
    rnn_cells = [
        tf.contrib.rnn.GRUCell(3),
        tf.contrib.rnn.GRUCell(3),
        tf.contrib.rnn.GRUCell(3),
    ]
    def m(env, rnn=tf.contrib.rnn.MultiRNNCell(rnn_cells)):
        outputs, state = tf.nn.dynamic_rnn(
            cell=rnn, inputs=env,
            sequence_length=[N_HISTS], dtype='float32')
        # shape of `outputs`: `[1, N_HISTS, 3]`
        a, b, _ = tf.unstack(
            tf.unstack(tf.squeeze(outputs), axis=0)[0]
        )  # scalars.
        return a, b
        
    optimizer = tf.train.RMSPropOptimizer(0.1)
    
    # Compute the meta-loss
    f = make_f(x)
    grad_f = tf.gradients(f, [x])[0]
    a, b = m(env)
    delta_x = a + b * grad_f
    meta_loss = make_f(x + delta_x)  # the value of `f` at the next step. If the
                                # model works well, it shall predict the
                                # next step that minimizes the value of f.
    
    # Update `w` with standard gradient descent optimizer
    train_op = optimizer.minimize(meta_loss)
    delta_f = meta_loss - f
    
    x_val = 100 * np.ones([1, n_dims], dtype='float32')
    env_val = np.zeros([1, N_HISTS, 3], dtype='float32')
    env_val[:,-1,:] = np.array([0.1, 0.1, 0])
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        time_start = time.time()
        
        for step in range(n_iters):
            
            delta_x_val, a_val, b_val, delta_f_val, f_val, _ = sess.run(
                [delta_x, a, b, delta_f, f, train_op],
                feed_dict={x: x_val, env: env_val}
            )
            print(step, f_val, delta_f_val)
            
            if (step+1) % N_SUB_TRAINS == 0:
                # Update `x` and `env`
                x_val += delta_x_val
                new_env_tail = np.asarray([[[a_val, b_val, delta_f_val]]])
                env_val = np.concatenate(
                    ( env_val[:,1:,:], new_env_tail ),
                    axis=1
                )
                
            if f_val < 1e-3:
                break
        
        elapsed_time = time.time() - time_start
        print('Elapsed time: {} sec.'.format(elapsed_time))
        # => Elapsed time: 25.550095081329346 sec.
       
        
        
        
if __name__ == '__main__':
    
    main()
