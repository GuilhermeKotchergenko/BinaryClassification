# Arquivo não modificado do checkpoint a fim de comparação
import autograd.numpy as np
from autograd import grad

class LogisticRegression:
    def __init__(self, lr=0.01, penalty=None, C=0.01, tolerance=1e-4, max_iters=1000):
        """Regressão logística binária com gradiente descendente.

        Parâmetros:
        - lr: taxa de aprendizado. Regula a velocidade de aprendizado do modelo.
        - penalty: None, 'l1(Lasso) ou 'l2'(Ridge). Força o modelo a ser mais simples e generalizar melhor.
        - C: coeficiente de regularização. Controla o quanto os pesos grandes são penalizados.
        - tolerance: critério de parada para o gradiente. Ajuda a parar o treino quando não há mais melhora significativa.
        - max_iters: número máximo de iterações. Garante que o treino não rode para sempre.
        """
        self.lr = lr
        self.penalty = penalty
        self.C = C
        self.tolerance = tolerance
        self.max_iters = max_iters
        
        self.theta = None
        self.errors = []
        
    def sigmoid(self, z):
        """Função sigmoide: transforma o valor linear em uma probabilidade entre 0 e 1."""
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        """Adiciona uma coluna de 1s no início de X para representar o termo de bias."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def _loss(self, w):
        """
        Calcula o custo da entropia cruzada com regularização (se houver).
        """
        z = np.dot(self.X, w)
        y_pred = self.sigmoid(z)
        
        # Evita log(0) com clipping
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        loss = -np.mean(self.y * np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred))

        # Adiciona penalização, ignorando o bias (posição 0)
        if self.penalty == 'l2':
            loss += (0.5 * self.C) * np.sum(w[1:] ** 2)
        elif self.penalty == 'l1':
            loss += self.C * np.sum(np.abs(w[1:]))

        return loss
    
    def fit(self, X, y):
        """
        Treina o modelo com gradiente descendente.
        """
        # Guarda os dados internamente
        self.X = self._add_intercept(X)
        self.y = y
        n_samples, n_features = self.X.shape

        # Inicializa os pesos (theta) com distribuição normal
        self.theta = np.random.normal(loc=0.0, scale=0.01, size=n_features)

        # Usa autograd para obter a derivada da função de custo
        gradient = grad(self._loss)

        # Lista para armazenar o histórico de perda
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
        """
        Retorna as probabilidades da classe positiva.
        """
        X = self._add_intercept(X)
        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.4):
        """
        Retorna as previsões binárias (0 ou 1) com base nas probabilidades.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)




        
        