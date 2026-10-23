import numpy as np


def Sigmoid(X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-X))


def dSigmoid(A: np.ndarray) -> np.ndarray:
    return A * (1 - A)


class Capa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas, neuronas) * 0.1
        self.bias = np.zeros((1, neuronas))
        self.A = None  # Activaciones
        self.X = None  # Entradas

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        self.A = Sigmoid(np.dot(X, self.pesos) + self.bias)
        return self.A

    def backpropagation(
        self,
        delta_prop: np.ndarray,
        pesos_siguiente: np.ndarray = None,
        lr: float = 0.1,
    ):
        if pesos_siguiente is not None:
            # Capa oculta
            delta = np.dot(delta_prop, pesos_siguiente.T) * dSigmoid(self.A)
        else:
            # Capa de salida
            delta = delta_prop * dSigmoid(self.A)

        # Gradientes de pesos y bias
        dW = np.dot(self.X.T, delta)
        dB = np.sum(delta, axis=0, keepdims=True)

        pesos_viejos = self.pesos.copy()

        # Actualizar parametros (SGD)
        self.pesos -= lr * dW
        self.bias -= lr * dB

        return delta, pesos_viejos


def main():
    X = np.ones((3, 2)) * 5
    y = np.zeros((3, 2))

    capa_oculta = Capa(entradas=2, neuronas=4)
    capa_salida = Capa(entradas=4, neuronas=2)

    epocas = 1000
    lr = 0.5

    print("--- DATOS INICIALES ---")
    print("X =\n", X)
    print("y deseada =\n", y, "\n")
    print("--- ENTRENAMIENTO ---")

    for epoca in range(epocas):
        # Forward
        A1 = capa_oculta.feedforward(X)
        Y_hat = capa_salida.feedforward(A1)  # Predict

        # MSE
        loss = 0.5 * np.mean((y - Y_hat) ** 2)

        # Back
        error_salida = Y_hat - y
        delta_salida, W_salida_viejos = capa_salida.backpropagation(error_salida, lr=lr)
        delta_oculta, _ = capa_oculta.backpropagation(
            delta_salida, W_salida_viejos, lr=lr
        )

        if epoca % 200 == 0 or epoca == epocas - 1:
            print(f"Época {epoca:4d} | Pérdida (MSE): {loss:.6f}")

    print("\n--- RESULTADO FINAL ---")
    print("Predicción final (Y_hat):\n", np.round(Y_hat, 4))


if __name__ == "__main__":
    main()
