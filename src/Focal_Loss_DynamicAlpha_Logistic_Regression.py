import autograd.numpy as np
from autograd import grad

class LogisticRegression:
    def __init__(self, lr=0.01, penalty=None, C=0.01, tolerance=1e-4, max_iters=1000, alpha=0.5, gamma=2.0):
        """Regressão logística binária com gradiente descendente e Focal Loss.

        Parâmetros:
        - lr: taxa de aprendizado. Regula a velocidade de aprendizado do modelo.
        - penalty: None, 'l1' (Lasso) ou 'l2' (Ridge). Força o modelo a ser mais simples e generalizar melhor.
        - C: coeficiente de regularização. Controla o quanto os pesos grandes são penalizados.
        - tolerance: critério de parada para o gradiente. Ajuda a parar o treino quando não há mais melhora significativa.
        - max_iters: número máximo de iterações. Garante que o treino não rode para sempre.
        - alpha: fator de ponderação da classe na Focal Loss.
        - gamma: fator de foco da Focal Loss. 
        """
        self.lr = lr
        self.penalty = penalty
        self.C = C
        self.tolerance = tolerance
        self.max_iters = max_iters
        self.alpha = alpha  # Novo hiperparâmetro
        self.gamma = gamma  # Novo hiperparâmetro
        
        self.theta = None
        self.errors = []
        
    def sigmoid(self, z):
        """Função sigmoide: transforma o valor linear em uma probabilidade entre 0 e 1."""
        z_clipped = np.clip(z, -30, 30)  # evita overflow em exp
        return 1 / (1 + np.exp(-z_clipped))
    
    def _add_intercept(self, X):
        """Adiciona uma coluna de 1s no início de X para representar o termo de bias."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _loss(self, w):
        """Calcula a Focal Loss com regularização."""
        z = np.dot(self.X, w)
        y_pred = self.sigmoid(z)
        
        # Evita log(0) com clipping
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        n_pos = np.sum(self.y)
        n_neg = len(self.y) - n_pos
        alpha_pos = n_neg / (n_pos + n_neg)   # quanto mais raros os positivos, maior α deles
        alpha_neg = n_pos / (n_pos + n_neg)

        pt = y_pred * self.y + (1 - y_pred) * (1 - self.y)
        alpha_t = self.y * alpha_pos + (1 - self.y) * alpha_neg

        focal_loss = -np.mean(alpha_t * (1 - pt) ** self.gamma * np.log(pt))

        # Adiciona penalização regularizada se necessário
        if self.penalty == 'l2':
            focal_loss += (0.5 * self.C) * np.sum(w[1:] ** 2)
        elif self.penalty == 'l1':
            focal_loss += self.C * np.sum(np.abs(w[1:]))

        return focal_loss
    
    def fit(self, X, y):
        """Treina o modelo com gradiente descendente."""
        self.X = self._add_intercept(X)
        self.y = y
        n_samples, n_features = self.X.shape

        # Inicializa os pesos (theta) com distribuição normal
        self.theta = np.random.normal(loc=0.0, scale=0.01, size=n_features)

        # Usa autograd para obter a derivada da função de custo
        gradient = grad(self._loss)
        self.errors = []

        for i in range(self.max_iters):
            grad_value = gradient(self.theta)
            self.theta -= self.lr * grad_value

            # Calcula o erro atual e salva
            error = self._loss(self.theta)
            self.errors.append(error)

            # Critério de parada (diferença entre erros consecutivos)
            if i > 0 and abs(self.errors[-2] - self.errors[-1]) < self.tolerance:
                print(f"Convergência alcançada em {i} iterações.")
                break
            
    def predict_proba(self, X):
        """Retorna as probabilidades da classe positiva."""
        X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        """Retorna as previsões binárias (0 ou 1) com base nas probabilidades."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
