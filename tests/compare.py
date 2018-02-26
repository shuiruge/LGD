#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Using standard optimizer, for comparing."""

import numpy as np
import tensorflow as tf
import time

from raw_implement import make_loss_and_gradients, n_dims, n_iters

tf.reset_default_graph()



def main(make_loss_and_gradients=make_loss_and_gradients,
         n_dims=n_dims, n_iters=n_iters):
    
    init_x = 100*np.ones([1, n_dims])
    x = tf.Variable(init_x, dtype='float32')
    loss, gradients = make_loss_and_gradients(x)
    optimizer = tf.train.RMSPropOptimizer(0.01)
    train_op = optimizer.minimize(loss)
    
    tf.summary.scalar('meta-loss', loss)
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        writer = tf.summary.FileWriter('../dat/logdir/2', sess.graph)
        
        sess.run(init)
        time_start = time.time()
        
        for step in range(n_iters):
            loss_val, summary_val, _ = sess.run([loss, summary_op, train_op])
            print(step, loss_val)
            
            writer.add_summary(summary_val, step)
            
            if loss_val < 1e-2:
                break
            
        elapsed_time = time.time() - time_start
        print('Elapsed time: {} sec.'.format(elapsed_time))
        # => Elapsed time: 8.54189419746399 sec.
        
        
            
if __name__ == '__main__':
    
    main()