import torch
class LinearModel:
    def __init__(self):
        self.w = None
    def score(self, X):
        if self.w is None:
            self.w = torch.rand((X.size(1),))
        return X @ self.w
    def predict(self, X):
        return (self.score(X) > 0).float()
class Perceptron(LinearModel):
    def loss(self, X, y):
        self.X = X  # Store for use in grad
        self.y = y
        y_mod = 2 * y - 1
        s = self.score(X)
        return (1.0 * (s * y_mod <= 0)).mean()
    def grad(self, X, y):
        y_hat = 2 * y - 1
        s = torch.mv(X, self.w)
        indicator = (s * y_hat < 0).float()
        grads = -indicator[:, None] * y_hat[:, None] * X
        return grads.mean(dim=0)


class PerceptronOptimizer():
    def __init__(self, model, alpha=1.0):  # default alpha = 1.0
        self.model = model
        self.alpha = alpha

    def step(self, x_i, y_i):
        self.model.x = x_i.squeeze()  # ensure x_i is 1D or 2D
        self.model.y = y_i.squeeze()
        _ = self.model.loss(x_i, y_i)
        grad = self.model.grad(x_i, y_i)
        self.model.w = self.model.w - self.alpha * grad  # <- scaled by alpha!



