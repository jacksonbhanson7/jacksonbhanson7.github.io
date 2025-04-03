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
    
class LogisticRegression(LinearModel):
    def loss(self, X, y):
        scores = self.score(X)
        sigma = torch.sigmoid(scores)  # σ(s_i)
        loss = -torch.mean(y * torch.log(sigma + 1e-9) + (1 - y) * torch.log(1 - sigma + 1e-9)) #add in tiny number so that we are never taking log(0)
        return loss
    
    def grad(self, X, y):    
        scores = self.score(X)                   
        sigma = torch.sigmoid(scores)
        v = sigma - y
        v_ = v[:, None]
        gradient = torch.mean(X * v_, dim=0)
        return gradient

class GradientDescentOptimizer:
    def __init__(self, model):
        self.model = model
        self.prev_w = None  # This will hold w_{k-1}

    def step(self, X, y, alpha, beta):
        # Store current weights
        current_w = self.model.w.clone()
        _ = self.model.loss(X, y)
        # Compute gradient of the loss at current weights
        grad = self.model.grad(X, y)
        # If no previous weights, treat it like regular gradient descent
        if self.prev_w is None:
            self.prev_w = current_w
        # Momentum term: (w_k - w_{k-1})
        momentum = beta * (current_w - self.prev_w)
        # Gradient descent step with momentum
        self.model.w = current_w - alpha * grad + momentum
        # Update prev_w for the next step
        self.prev_w = current_w
