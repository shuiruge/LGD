# LGD
Learning gradient descent (by gradient descent).

Explicitly, let `f(x)` a smooth function over Euclidean space, and suppose we are to find the `argmin(f)` using gradient descent algorithm with momentum, we update `x` at each iteration by

  g = compute_gradient(f, x)
  delta_x = a + b * g
  x = x - delta_x
  
wherein the hyper-parameter `a` (momentum) and `b` (learning-rate) are fixed. But now, we are trying to upgrade this, by setting `a` and `b` functions over all the past values of `a`, `b`, and the difference between the adjoint values of `f` along the iteration. We parameterize these unknown functions by neural network, with the utility of its universality. And then train the weights and biases of the neural network by the standard (human-designed) gradient descent algorithm(s).

This is an experiment.
