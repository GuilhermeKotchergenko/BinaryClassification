import autograd.numpy as np
from autograd import grad

class BCE_Logistic_Sigmoid:
    def __init__(self, lr=0.001, tolerance=1e-6, max_iters=1000, temperature=1.2):
        self.lr = lr
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.temperature = temperature
        self.theta = None
        self.errors = []

    def sigmoid(self, z):
        """Sigmoide com fator de temperatura"""
        z_clipped = np.clip(z, -30, 30)
        return 1 / (1 + np.exp(-z_clipped / self.temperature))

    def _add_intercept(self, X):
        """Adiciona um intercepto ao conjunto de features"""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def _loss(self, w):
        z = np.dot(self.X, w)
        y_pred = self.sigmoid(z)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(self.y * np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred))
        return loss

    def fit(self, X, y):
        self.X = self._add_intercept(X)
        self.y = y
        n_samples, n_features = self.X.shape

        np.random.seed(42)  # Garantir reprodutibilidade
        self.theta = np.random.normal(loc=0.0, scale=0.01, size=n_features)

        gradient = grad(self._loss)
        self.errors = []

        for i in range(self.max_iters):
            grad_value = gradient(self.theta)
            self.theta -= self.lr * grad_value

            error = self._loss(self.theta)
            self.errors.append(error)

            if i > 0 and abs(self.errors[-2] - self.errors[-1]) < self.tolerance:
                print(f"Convergência alcançada em {i} iterações.")
                break

    def predict_proba(self, X):
        X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
