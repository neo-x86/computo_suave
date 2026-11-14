import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(z, -50, 50))
    return exp / exp.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, arquitectura, tarea, semilla):
        rng = np.random.default_rng(semilla)
        self.tarea = tarea
        self.pesos, self.bias = [], []
        for entradas, salidas in zip(arquitectura[:-1], arquitectura[1:]):
            limite = np.sqrt(2.0 / entradas)
            self.pesos.append(rng.normal(0, limite, (entradas, salidas)))
            self.bias.append(np.zeros((1, salidas)))

    def forward(self, X):
        activaciones = [X]
        preactivaciones = []
        A = X
        for i, (W, b) in enumerate(zip(self.pesos, self.bias)):
            Z = A @ W + b
            preactivaciones.append(Z)
            if i == len(self.pesos) - 1:
                A = softmax(Z) if self.tarea == "clasificacion" else Z
            else:
                A = np.maximum(Z, 0)
            activaciones.append(A)
        return activaciones, preactivaciones

    def predecir(self, X):
        return self.forward(X)[0][-1]

    def entrenar(self, X, y, epocas, lr, batch, rng):
        historial = []
        for _ in range(epocas):
            perdidas = []
            orden = rng.permutation(len(X))
            for inicio in range(0, len(X), batch):
                idx = orden[inicio:inicio + batch]
                xb, yb = X[idx], y[idx]
                activaciones, zs = self.forward(xb)
                salida = activaciones[-1]
                if self.tarea == "clasificacion":
                    perdida = -np.mean(np.sum(yb * np.log(salida + 1e-12), axis=1))
                    delta = salida - yb
                else:
                    perdida = np.mean((salida - yb) ** 2)
                    delta = 2 * (salida - yb) / salida.shape[1]
                perdidas.append(float(perdida))
                for i in range(len(self.pesos) - 1, -1, -1):
                    dW = activaciones[i].T @ delta / len(xb)
                    db = delta.mean(axis=0, keepdims=True)
                    anterior = self.pesos[i].copy()
                    self.pesos[i] -= lr * dW
                    self.bias[i] -= lr * db
                    if i:
                        delta = (delta @ anterior.T) * (zs[i - 1] > 0)
            historial.append(float(np.mean(perdidas)))
        return historial
