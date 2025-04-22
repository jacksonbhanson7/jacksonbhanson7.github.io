import torch

class LinearModel:
    def __init__(self):
        """
        Initializes the model with no weights.
        """
        self.w = None



    def predict(self, X):
        """
        Generates yes/no predictions using scores

        Args:
            X (torch.Tensor): Input feature matrix of shape n_samples x n_features

        Returns:
            torch.Tensor: Predicted yes/no class labels (0 or 1)
        """
        return (self.score(X) > 0).float()


class SparseKernelLogisticRegression(LinearModel):
    
    def __init__(self, kernel_fn, lam, gamma):
        self.kernel_fn = kernel_fn
        self.lam = lam
        self.gamma = gamma

    
    def score(self, X):
        # Compute kernel between new inputs and stored training data
        K = self.kernel_fn(X, self.X_train, self.gamma)
        return K @ self.a
    
    def loss(self, X, y):
        s = self.score(X)
        sigma = torch.sigmoid(s)
        log_likelihood = -torch.mean(y * torch.log(sigma + 1e-9) + (1 - y) * torch.log(1 - sigma + 1e-9))
        l1_penalty = self.lam * torch.norm(self.a, p=1)
        return log_likelihood + l1_penalty

    def grad(self, X, y):
        K = self.kernel_fn(X, self.X_train, self.gamma)
        sigma = torch.sigmoid(self.score(X))
        grad = K.T @ (sigma - y) / y.shape[0]
        grad += self.lam * torch.sign(self.a)
        return grad
    
    def fit(self, X, y, m_epochs=50000, lr=0.001):
        """
        Trains the kernel logistic regression model.

        Args:
            X (torch.Tensor): Training feature matrix
            y (torch.Tensor): Training labels
            m_epochs (int): Number of training iterations
            lr (float): Learning rate
        """
        self.X_train = X
        self.a = torch.zeros(X.shape[0])  # One coefficient per training example

        for _ in range(m_epochs):
            grad = self.grad(X, y)
            self.a -= lr * grad