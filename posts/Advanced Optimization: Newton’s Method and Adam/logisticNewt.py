import torch

class LinearModel:
    def __init__(self):
        """
        Initializes the model with no weights.
        """
        self.w = None

    def score(self, X):
        """
        Computes the scores and if weights are not initialized, initialize them randomly

        Args:
            X (torch.Tensor): Input feature matrix of shape n_samples x n_features

        Returns:
            torch.Tensor: scores of shape n_samples,
        """
        if self.w is None:
            self.w = torch.rand((X.shape[1],))
        return X @ self.w

    def predict(self, X):
        """
        Generates yes/no predictions using scores

        Args:
            X (torch.Tensor): Input feature matrix of shape n_samples x n_features

        Returns:
            torch.Tensor: Predicted yes/no class labels (0 or 1)
        """
        return (self.score(X) > 0).float()


class LogisticRegression(LinearModel):
    def loss(self, X, y):
        """
        Computes the binary cross-entropy loss for logistic regression.

        Args:
            X (torch.Tensor): Input features
            y (torch.Tensor): Actual yes/no class labels

        Returns:
            torch.Tensor: loss value
        """
        scores = self.score(X)
        sigma = torch.sigmoid(scores)
        # Add tiny, tiny number to avoid to avoid log(0)
        loss = -torch.mean(y * torch.log(sigma + 1e-9) + (1 - y) * torch.log(1 - sigma + 1e-9))
        return loss

    def grad(self, X, y):
        """
        Computes the gradient of the binary cross-entropy with calcualted weights.

        Args:
            X (torch.Tensor): Input features matrix
            y (torch.Tensor): Actual yes/no class labels

        Returns:
            torch.Tensor: Gradient vector of shape n_features,
        """
        scores = self.score(X)
        sigma = torch.sigmoid(scores)
        gradient = torch.mean((sigma - y).unsqueeze(1) * X, dim=0)
        return gradient

    
    def hessian(self, X):
        """
        Compute the Hessian matrix of the loss function at the current weights.
        Args:
            X: Tensor of shape (n_samples, n_features), the input data.
        Returns:
            Tensor of shape (n_features, n_features), the Hessian matrix.
        """
        s = self.score(X)  # shape (n_samples,)
        sigma = torch.sigmoid(s)  # shape (n_samples,)
        D = sigma * (1 - sigma)  # shape (n_samples,)
        D_mat = torch.diag(D)    # create diagonal matrix
        H = X.T @ D_mat @ X / X.shape[0]
        return H


class GradientDescentOptimizer:
    def __init__(self, model, alpha):
        """
        Initialize optimizer for a given model.

        Args:
            model (LinearModel): The model whose parameters will be optimized
        """
        self.model = model
        self.prev_w = None  # To store previous weights for momentum
        self.alpha = alpha

    def step(self, X, y, beta):
        """
        Perform one step of gradient descent with optional momentum.

        Args:
            X (torch.Tensor): Training features
            y (torch.Tensor): Training labels
            alpha (float): Learning rate
            beta (float): Momentum factor (0 = no momentum)
        """
        current_w = self.model.w.clone()
        grad = self.model.grad(X, y)

        if self.prev_w is None:
            self.prev_w = current_w

        momentum = beta * (current_w - self.prev_w)
        self.model.w = current_w - self.alpha * grad + momentum
        self.prev_w = current_w



class NewtonOptimizer:
    def __init__(self, model, alpha=1.0):
        self.model = model
        self.alpha = alpha

    def step(self, X, y):
        grad = self.model.grad(X, y)
        hess = self.model.hessian(X)
        eps = 1e-4  # Small regularization strength
        hess += eps * torch.eye(hess.shape[0])  # Add small multiple of identity
        update_direction = torch.linalg.solve(hess, grad)
        self.model.w -= self.alpha * update_direction




class AdamOptimizer:
    """
    Optimizer that implements the Adam algorithm for stochastic optimization.
    
    Parameters
    ----------
    model : LogisticRegression
        The model whose parameters will be optimized.
    batch_size : int
        Size of the minibatches to use in each step.
    alpha : float
        Step size (learning rate).
    beta_1 : float
        Exponential decay rate for the first moment estimates.
    beta_2 : float
        Exponential decay rate for the second moment estimates.
    w_0 : torch.Tensor, optional
        Initial value for the weights. If None, they will be initialized randomly.
    """
    def __init__(self, model, batch_size, alpha=0.001, beta_1=0.9, beta_2=0.999, w_0=None):
        self.model = model
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-8
        self.t = 0  # timestep
        self.w_0 = w_0  # store for delayed initialization
        self.m = None  # first moment vector
        self.v = None  # second moment vector

    def step(self, X, y):
        """
        Perform one step of the Adam optimization algorithm on a random minibatch.

        Parameters
        ----------
        X : torch.Tensor
            The input features.
        y : torch.Tensor
            The binary target labels (0 or 1).
        """
        if self.model.w is None:
            if self.w_0 is not None:
                self.model.w = self.w_0
            else:
                self.model.w = torch.randn(X.shape[1], 1) * 0.01

        # Lazy init moment vectors once model.w is available
        if self.m is None or self.v is None:
            self.m = torch.zeros_like(self.model.w)
            self.v = torch.zeros_like(self.model.w)

        # Increment timestep
        self.t += 1

        # Sample minibatch
        idx = torch.randperm(X.shape[0])[:self.batch_size]
        X_batch = X[idx]
        y_batch = y[idx]

        # Gradient
        grad = self.model.grad(X_batch, y_batch)

        # First and second moment estimates
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad ** 2)

        # Bias correction
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        # Update weights
        self.model.w -= self.alpha * m_hat / (torch.sqrt(v_hat) + self.epsilon)

