#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Using standard optimizer, for comparing."""

import numpy as np
import tensorflow as tf
import time

from raw_implement import make_f, n_dims, n_iters

tf.reset_default_graph()



def main(make_f=make_f, n_dims=n_dims, n_iters=n_iters):
    
    init_x = 100*np.ones([1, n_dims])
    x = tf.Variable(init_x, dtype='float32')
    f = make_f(x)
    optimizer = tf.train.RMSPropOptimizer(0.01)
    train_op = optimizer.minimize(f)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        
        sess.run(init)
        
        time_start = time.time()
        
        for step in range(n_iters):
            f_val, _ = sess.run([f, train_op])
            print(step, f_val)
            
            if f_val < 1e-3:
                break
            
        elapsed_time = time.time() - time_start
        print('Elapsed time: {} sec.'.format(elapsed_time))
        # => Elapsed time: 8.54189419746399 sec.
        
        
            
if __name__ == '__main__':
    
    main()