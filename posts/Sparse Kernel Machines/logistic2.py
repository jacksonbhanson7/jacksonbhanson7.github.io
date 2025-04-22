import torch
class SparseKernelLogisticRegression():
    
    def __init__(self, kernel_fn, lam, gamma):
        """
        Initializes the sparse kernel logistic regression model.

        Args:
            kernel_fn (callable): A function that computes the kernel between two input matrices.
            lam (float): Regularization strength for L1 penalty.
            gamma (float): Parameter for the kernel function (e.g., RBF bandwidth).
        """
        self.kernel_fn = kernel_fn
        self.lam = lam
        self.gamma = gamma

    
    def score(self, X):
        """
        Computes the kernelized score for new input data.

        Args:
            X (torch.Tensor): Input feature matrix of shape (n_samples, n_features).

        Returns:
            torch.Tensor: Vector of predicted scores.
        """
        
        K = self.kernel_fn(X, self.X_train, self.gamma)
        return K @ self.a
    
    def loss(self, X, y):
        """
        Computes the L1-regularized logistic loss.

        Args:
            X (torch.Tensor): Input feature matrix.
            y (torch.Tensor): Binary target labels (0 or 1).

        Returns:
            torch.Tensor: Scalar loss value.
        """

        s = self.score(X)
        sigma = torch.sigmoid(s)
        log_likelihood = -torch.mean(y * torch.log(sigma + 1e-9) + (1 - y) * torch.log(1 - sigma + 1e-9))
        l1_penalty = self.lam * torch.norm(self.a, p=1)
        return log_likelihood + l1_penalty

    def grad(self, X, y):
        """
        Computes the gradient of the regularized logistic loss.

        Args:
            X (torch.Tensor): Input feature matrix.
            y (torch.Tensor): Binary target labels.

        Returns:
            torch.Tensor: Gradient with respect to weight vector a.
        """
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
    
    def predict(self, X):
        """
        Generates yes/no predictions using scores

        Args:
            X (torch.Tensor): Input feature matrix of shape n_samples x n_features

        Returns:
            torch.Tensor: Predicted yes/no class labels (0 or 1)
        """
        return (self.score(X) > 0).float()