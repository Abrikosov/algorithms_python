import collections

def gradient(f, x0):
    """Returns value of gradient."""
    # General information:
    # https://en.wikipedia.org/wiki/Numerical_differentiation
    # https://en.wikipedia.org/wiki/Gradient

    # h = x * sqrt(epsilon), if x<0 then h<0 but for this implementation it is ok.
    h = lambda x: 1e-8 if x == 0.0 else x * 1e-8

    grad = None

    # is Iterable
    if isinstance(x0, collections.Iterable):
        grad = []
        for i in range(len(x0)):
            dx = h(x0[i])
            # copy list without reference
            x1 = x0[:]
            x1[i] -= dx
            x2 = x0[:]
            x2[i] += dx
            grad.append((f(*x2) - f(*x1)) / (2*dx))
    else:
        dx = h(x0)
        grad = (f(x0 + dx) - f(x0 - dx)) / (2*dx)

    return grad

def gradient_descent_for_function(f, x0, step_size = 0.01, maxiter=10000, x_tol = 1e-5):
    """Returns local minimum of a function one variable and appropriate x."""
    # General information:
    # https://en.wikipedia.org/wiki/Gradient_descent
    x = x0
    for i in range(maxiter):
        dx = - step_size * gradient(f, x)
        x += dx
        if abs(dx) < x_tol:
            return {"f": f(x), "x": x}
    raise RuntimeError("Maximum number of function evaluations has been exceeded.")

if __name__ == "__main__":
    pass