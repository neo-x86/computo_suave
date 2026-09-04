#!/usr/bin/env python3
import numpy as np

def ReLU(X: np.ndarray) -> np.ndarray:
    return np.maximum(0, X)

class Capa:
    def __init__(self, entradas: np.ndarray, neuronas: int,
                 activacion, feedforward, backpropagation, bias: bool = True) -> None:
        self.neuronas = neuronas
        self.activacion = activacion
        self.bias = bias
        self.feedforward = feedforward
        self.backpropagation = backpropagation
        self.entradas = entradas

        caracteristicas = entradas.shape[1]
        filas = caracteristicas + 1 if bias else caracteristicas
        self.pesos = np.random.randn(filas, neuronas)
        # self.pesos = np.array([[0, 0, 0.1, 0.2],
        #       [0.3, -1, 0.4, 0.5],
        #       [1, 0.6, 0.7, 0.8]])

    def getPesos(self) -> np.ndarray:
        return self.pesos

    def getResult(self) -> np.ndarray:
        return self.activacion(self.entradas @ self.pesos)


def genRandX(rows: int, cols: int) -> np.ndarray:
    return np.random.randn(rows, cols)

def main():
    X = genRandX(2, 3)
    # X = np.array([[1,2,3],[4,5,6]])
    capa = Capa(entradas=X, neuronas=1, activacion=ReLU,
                feedforward=None, backpropagation=None, bias=False)

    print("X=", X, end="\n\n")
    print("W=", capa.getPesos(), end="\n\n")
    print("Z=", capa.getResult(), end="\n\n")

if __name__ == "__main__":
    main()
