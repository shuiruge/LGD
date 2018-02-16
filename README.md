# LGD
Learning gradient descent (by gradient descent). (This is an experiment.)


### Idea

Let `f(x)` a smooth function over Euclidean space, and suppose we are to find the `argmin(f)` using gradient descent algorithm with momentum, we update `x` at each iteration by

    g = compute_gradient(f, x)
    delta_x = a + b * g
    x = x - delta_x
  
wherein the hyper-parameter `a` (momentum) and `b` (learning-rate) are fixed. But now, we are trying to upgrade this, by setting `a` and `b` functions over all the past values of `a`, `b`, and the difference between the adjoint values of `f` along the iteration. We parameterize these unknown functions by neural network, with the utility of its universality. And then train the weights and biases of the neural network by the standard (human-designed) gradient descent algorithm(s).


### Algorithm

    # Parameter
    N_SUB_TRAINS = ...

    # Variables
    E = Variable(..., trainable=False)  # "environment".
    x = Variable(..., trainable=False)
    w = Variable(..., trainable=True)  # weights and bias of the model below.

    # Model
    def m(E, w): ...  # maybe a RNN

    for step in range(n_iters):

        # Compute the meta-loss
        f = f(x)
        grad_f = compute_gradient(f, x)
        a, b = m(E, w)
        delta_x = a + b * grad_f
        meta_loss = f(x + delta_x)  # the value of `f` at the next step. If the
                                    # model works well, it shall predict the
                                    # next step that minimizes the value of f.

        # Update `w` with standard gradient descent optimizer
        grad_meta_loss = compute_gradient(meta_loss, w)
        Optimizer.apply_gradient(grad_meta_loss)

        if (step+1) % N_SUB_TRAINS == 0:
            # Upate the non-traiable variables
            x <- x + delta_x
            e = (a, b, ...)
            E <- E.append(e)


### Cost

Let `Dx` the dimension of `x`, `Dw` the dimension of `w`, and `Cf` the
complexity of `f`, and `Cm` of `m`. Within the iteration:

    for step in range(n_iters):

        # Compute the meta-loss
        f = f(x)  # costs nothing, computed in the previous iteration.
        grad_f = compute_gradient(f, x)  # Computed in the prevous iteration
                                         # in `compute_gradient(meta_loss, w)`.
        a, b = m(E, w)  # Theta of Cm.
        delta_x = a + b * grad_f  # Theta of Dx.
        meta_loss = f(x + delta_x)  # Theta of Cf.

        grad_meta_loss = compute_gradient(meta_loss, w)  # Theta of (Cf + Cm
                                                         # + Dx + Dw)
        Optimizer.apply_gradient(grad_meta_loss)

        if (step+1) % N_SUB_TRAINS == 0:
            # Upate the non-traiable variables
            x <- x + delta_x
            e = (a, b, ...)
            E <- E.append(e)

Thus, the total temporal cost is Theta of
    
    (2*Cm + 2*Cf + 2*Dx + Dw) * n_iters

while the standard optimization takes Theta of

    2 * Cf * n_iters

So, if `Cf` is much greater than `Cm`, then this method matters.

PS: grad(meta_loss, w)(w) = sum(
        grad(f, x)(x + delta_x) * ( grad(a, w)(w)
            + grad(b, w)(w) * grad(f, x)(x) ),
        x)
