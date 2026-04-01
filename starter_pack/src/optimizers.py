import numpy as np


# -------------------------
# SGD
# -------------------------
def sgd_update(param, grad, lr):
    return param - lr * grad


# -------------------------
# Momentum
# -------------------------
def init_momentum_state():
    return {}


def momentum_update(param, grad, lr, velocity, beta=0.9):
    velocity = beta * velocity - lr * grad
    param = param + velocity
    return param, velocity


# -------------------------
# Adam
# -------------------------
def init_adam_state():
    return {}


def adam_update(param, grad, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v