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


class GradientDescentOptimizer:
    def __init__(self, model):
        """
        Initialize optimizer for a given model.

        Args:
            model (LinearModel): The model whose parameters will be optimized
        """
        self.model = model
        self.prev_w = None  # To store previous weights for momentum

    def step(self, X, y, alpha, beta):
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
        self.model.w = current_w - alpha * grad + momentum
        self.prev_w = current_w
