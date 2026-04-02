import numpy as np


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, lr):
        self.lr = lr

    def update(self, param, grad):
        """Update parameter with gradient."""
        return param - self.lr * grad


class Momentum:
    """Momentum optimizer with velocity accumulation."""

    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = {}

    def init_velocity(self, param_id):
        """Initialize velocity for a parameter."""
        self.velocity[param_id] = 0.0

    def update(self, param, grad, param_id):
        """Update parameter with momentum."""
        if param_id not in self.velocity:
            self.init_velocity(param_id)

        self.velocity[param_id] = self.beta * self.velocity[param_id] - self.lr * grad
        param = param + self.velocity[param_id]
        return param

    def reset(self):
        """Reset velocity state."""
        self.velocity = {}


class Adam:
    """Adam optimizer with adaptive learning rates."""

    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def init_moments(self, param_id):
        """Initialize first and second moment estimates."""
        self.m[param_id] = 0.0
        self.v[param_id] = 0.0

    def update(self, param, grad, param_id):
        """Update parameter with adaptive learning rate."""
        if param_id not in self.m:
            self.init_moments(param_id)

        self.t += 1
        m = self.beta1 * self.m[param_id] + (1 - self.beta1) * grad
        v = self.beta2 * self.v[param_id] + (1 - self.beta2) * (grad ** 2)

        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)

        param = param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        self.m[param_id] = m
        self.v[param_id] = v

        return param

    def reset(self):
        """Reset optimizer state."""
        self.m = {}
        self.v = {}
        self.t = 0


# Backward compatibility functions (deprecated)
def sgd_update(param, grad, lr):
    return param - lr * grad

def init_momentum_state():
    return {}

def momentum_update(param, grad, lr, velocity, beta=0.9):
    velocity = beta * velocity - lr * grad
    param = param + velocity
    return param, velocity

def init_adam_state():
    return {}

def adam_update(param, grad, lr, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v