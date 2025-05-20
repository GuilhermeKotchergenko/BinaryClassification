import autograd.numpy as np
from autograd import grad

class LogisticRegression:
    def __init__(self, lr=0.01, penalty=None, C=0.01, tolerance=1e-4, max_iters=1000, temperature=1.5, threshold=0.4):
        self.lr = lr
        self.penalty = penalty
        self.C = C
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.temperature = temperature
        self.threshold = threshold
        
        self.theta = None
        self.errors = []
    
    def sigmoid(self, z):
        """Função sigmoide com fator de temperatura."""
        z_clipped = np.clip(z, -30, 30)
        return 1 / (1 + np.exp(-z_clipped / self.temperature))

    def _add_intercept(self, X):
        """Adiciona uma coluna de 1s no início de X para representar o termo de bias."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def calculate_class_weights(self, y):
        pos_weight = len(y) / (2 * sum(y))
        neg_weight = len(y) / (2 * (len(y) - sum(y)))
        return neg_weight, pos_weight

    def _loss(self, w):
        z = np.dot(self.X, w)
        y_pred = self.sigmoid(z)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Calcular pesos automaticamente
        neg_weight, pos_weight = self.calculate_class_weights(self.y)

        # Loss ponderada
        weighted_loss = -np.mean(
            pos_weight * self.y * np.log(y_pred) +
            neg_weight * (1 - self.y) * np.log(1 - y_pred)
        )
        
        return weighted_loss

    def fit(self, X, y):
        """Treina o modelo com gradiente descendente."""
        self.X = self._add_intercept(X)
        self.y = y
        n_samples, n_features = self.X.shape

        # Inicializa os pesos (theta) com distribuição normal
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
        """Retorna as probabilidades da classe positiva."""
        X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        """Retorna as previsões binárias (0 ou 1) com base no threshold ajustado."""
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
