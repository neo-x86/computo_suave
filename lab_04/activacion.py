#!/usr/bin/env python3

import numpy as np


def Sigmoid(X: np.ndarray) -> np.ndarray:
    # clip para evitar overflow en exp() con entradas grandes
    return 1 / (1 + np.exp(-np.clip(X, -500, 500)))


def dSigmoid(A: np.ndarray) -> np.ndarray:
    return A * (1 - A)


def Softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)
