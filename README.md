# LGD
Learning gradient descent (by gradient descent). (This is an experiment.)


### Idea

Let `f(x)` a smooth function over Euclidean space, and suppose we are to find the `argmin(f)` using gradient descent algorithm with momentum, we update `x` at each iteration by

    g = compute_gradient(f, x)
    delta_x = a + b * g
    x = x - delta_x
  
wherein the hyper-parameter `a` (momentum) and `b` (learning-rate) are fixed. But now, we are trying to upgrade this, by setting `a` and `b` functions over all the past values of `a`, `b`, and the difference between the adjoint values of `f` along the iteration. We parameterize these unknown functions by neural network, with the utility of its universality. And then train the weights and biases of the neural network by the standard (human-designed) gradient descent algorithm(s).


### Algorithm

    # Variables
    E = Variable(..., trainable=False)  # "environment".
    x = Variable(..., trainable=False)
    w = Variable(..., trainable=True)  # weights and bias of the model below.

    # Model
    def m(E, w): ...  # maybe an RNN

    for step in range(n_iters):

        # Compute the meta-loss
        f = f(x)
        grad_f = compute_gradient(f, x)
        a, b = m(E, w)
        delta_x = a + b * grad_f
        meta_loss = f(x + delta_x)  # the value of `f` at the next step. If the
                                    # model works well, it shall predict the
                                    # next step that minimizes this value.

        # Update `w` with standard gradient descent optimizer
        grad_meta_loss = compute_gradient(meta_loss, w)
        Optimizer.apply_gradient(grad_meta_loss)

        # Upate the non-traiable variables
        x <- x + delta_x
        E <- E.append(a, b, ...)





