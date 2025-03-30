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
        y_hat = 2 * y - 1                       # shape (k,)
        scores = torch.mv(X, self.w)           # shape (k,)
        indicator = (scores * y_hat < 0).float()  # shape (k,)

        # reshape so indicator * y_hat becomes (k, 1) and broadcasts with X
        weights = (indicator * y_hat).unsqueeze(1)  # shape (k, 1)
        grad = -torch.mean(weights * X, dim=0)      # shape (p,)
        return grad

class PerceptronOptimizer():
    def __init__(self, model, alpha=1.0):  # default alpha = 1.0
        self.model = model
        self.alpha = alpha

    def step(self, X, y):
        self.model.X = X
        self.model.y = y
        _ = self.model.loss(X, y)
        grad = self.model.grad(X, y)
        self.model.w = self.model.w - self.alpha * grad
