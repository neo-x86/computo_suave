#!/usr/bin/env python3

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from activacion import Sigmoid, dSigmoid, Softmax

PATH_CSV = "./data/rt_iot2022/RT_IOT2022.csv"

SEMILLA = 42
COLUMNA_OBJETIVO = "Attack_type"
COLUMNAS_CATEGORICAS = ["proto", "service"]


class Capa:
    def __init__(self, entradas: int, neuronas: int, activacion: str = "sigmoid"):
        # Inicialización tipo Xavier: evita saturar la sigmoide con muchas entradas
        self.pesos = np.random.randn(entradas, neuronas) * np.sqrt(1.0 / entradas)
        self.bias = np.zeros((1, neuronas))
        self.activacion = activacion
        self.A = None  # Activaciones
        self.X = None  # Entradas

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        Z = np.dot(X, self.pesos) + self.bias
        self.A = Softmax(Z) if self.activacion == "softmax" else Sigmoid(Z)
        return self.A

    def backpropagation(
        self,
        delta_prop: np.ndarray,
        pesos_siguiente: np.ndarray = None,
        lr: float = 0.1,
    ):
        if pesos_siguiente is not None:
            delta = np.dot(delta_prop, pesos_siguiente.T) * dSigmoid(self.A)
        elif self.activacion == "softmax":
            delta = delta_prop
        else:
            delta = delta_prop * dSigmoid(self.A)

        # Gradientes de pesos y bias (promediados sobre el mini-lote)
        n = self.X.shape[0]
        dW = np.dot(self.X.T, delta) / n
        dB = np.sum(delta, axis=0, keepdims=True) / n

        pesos_viejos = self.pesos.copy()

        # Actualizar parametros (SGD)
        self.pesos -= lr * dW
        self.bias -= lr * dB

        return delta, pesos_viejos


class RedNeuronal:
    """Perceptrón multicapa formado por una lista de objetos Capa."""

    def __init__(self, tamanos: list):
        n = len(tamanos) - 1
        self.capas = [
            Capa(tamanos[i], tamanos[i + 1], "softmax" if i == n - 1 else "sigmoid")
            for i in range(n)
        ]

    def predecir(self, X: np.ndarray) -> np.ndarray:
        A = X
        for capa in self.capas:
            A = capa.feedforward(A)
        return A

    def entrenar_lote(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        Y_hat = self.predecir(X)
        # Cross-entropy
        loss = -np.mean(np.sum(y * np.log(Y_hat + 1e-12), axis=1))

        # Backpropagation de la última capa a la primera
        delta = Y_hat - y
        pesos_sig = None
        for capa in reversed(self.capas):
            delta, pesos_viejos = capa.backpropagation(delta, pesos_sig, lr=lr)
            pesos_sig = pesos_viejos
        return loss

    def entrenar(
        self, X, y, epocas, lr, tam_lote, X_val=None, y_val=None, verbose=True
    ):
        n = X.shape[0]
        rng = np.random.default_rng(SEMILLA)
        historial = []
        for epoca in range(epocas):
            idx = rng.permutation(n)
            perdidas = []
            for i in range(0, n, tam_lote):
                lote = idx[i : i + tam_lote]
                perdidas.append(self.entrenar_lote(X[lote], y[lote], lr))
            loss = float(np.mean(perdidas))
            historial.append(loss)

            if verbose and (epoca % max(1, epocas // 10) == 0 or epoca == epocas - 1):
                acc_tr = accuracy_score(y.argmax(1), self.predecir(X).argmax(1))
                msg = f"Epoch {epoca:4d} | Loss (Cross-Entropy): {loss:.6f} | Acc train: {acc_tr:.4f}"
                if X_val is not None:
                    acc_te = accuracy_score(
                        y_val.argmax(1), self.predecir(X_val).argmax(1)
                    )
                    msg += f" | Acc test: {acc_te:.4f}"
                print(msg)
        return historial


def cargar_datos(ruta):
    df = pd.read_csv(ruta)

    if df.columns[0] == "" or str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=df.columns[0])

    df = df.dropna().reset_index(drop=True)

    y_texto = df[COLUMNA_OBJETIVO].astype(str)
    X_df = df.drop(columns=[COLUMNA_OBJETIVO])

    # Variables categóricas (proto, service) -> one-hot
    cats = [c for c in COLUMNAS_CATEGORICAS if c in X_df.columns]
    X_df = pd.get_dummies(X_df, columns=cats, dtype=float)

    return X_df, y_texto


def resumen_datos(X_df: pd.DataFrame, y_texto: pd.Series):
    print("--- DESCRIPCIÓN DEL DATASET ---")
    print(
        f"Registros: {len(X_df)} | Variables de entrada (tras one-hot): {X_df.shape[1]}"
    )
    print("Variable a predecir: Attack_type")
    print("Distribución de clases:")
    for clase, cuenta in y_texto.value_counts().items():
        print(f"  {clase:32s} {cuenta:7d} ({100 * cuenta / len(y_texto):5.2f}%)")
    print()


def evaluar(nombre, y_real, probs, clases):
    """y_real: índices de clase; probs: salidas de la red (n, n_clases)."""
    y_pred = probs.argmax(1)
    print(f"[{nombre}]")
    print(f"  Accuracy                 : {accuracy_score(y_real, y_pred):.4f}")
    print(
        f"  Precision (macro)        : "
        f"{precision_score(y_real, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        f"  Precision (weighted)     : "
        f"{precision_score(y_real, y_pred, average='weighted', zero_division=0):.4f}"
    )

    # ROC-AUC one-vs-rest
    presentes = np.unique(y_real)
    if len(presentes) < 2:
        print("  ROC-AUC (OvR, macro)     : N/A (solo hay una clase en estos datos)")
        return
    P = probs[:, presentes]
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)  # filas suman 1
    try:
        auc = roc_auc_score(
            y_real, P, multi_class="ovr", average="macro", labels=presentes
        )
        print(f"  ROC-AUC (OvR, macro)     : {auc:.4f}")
    except ValueError as e:
        print(f"  ROC-AUC (OvR, macro)     : N/A ({e})")


def main():
    np.random.seed(SEMILLA)

    ruta = PATH_CSV
    X_df, y_texto = cargar_datos(ruta)
    resumen_datos(X_df, y_texto)

    # Codificar etiquetas -> índices y one-hot
    clases = np.array(sorted(y_texto.unique()))
    y_idx = np.searchsorted(clases, y_texto.values)
    Y = np.eye(len(clases))[y_idx]
    X = X_df.values.astype(float)

    # Partición 60/40 (entrenamiento/prueba). Se estratifica si todas las clases
    # tienen al menos 2 registros (con muestras muy pequeñas no es posible).
    conteos = np.bincount(y_idx)
    estratificar = y_idx if conteos.min() >= 2 and len(clases) > 1 else None
    X_train, X_test, Y_train, Y_test, yi_train, yi_test = train_test_split(
        X, Y, y_idx, test_size=0.4, random_state=SEMILLA, stratify=estratificar
    )
    print(f"Train: {len(X_train)} registros | Test: {len(X_test)} registros (60/40)\n")

    # Estandarización (ajustada SOLO con entrenamiento para no filtrar información)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Arquitectura
    tamanos = [X_train.shape[1], 64, 32, len(clases)]
    red = RedNeuronal(tamanos)
    print("--- ARQUITECTURA ---")
    print(
        " -> ".join(str(t) for t in tamanos), "(ocultas: sigmoide, salida: softmax)\n"
    )

    epocas = 100
    lr = 0.5
    tam_lote = 128

    print("--- TRAINING ---")
    red.entrenar(X_train, Y_train, epocas, lr, tam_lote, X_test, Y_test)

    print("\n--- RESULTADOS ---")
    evaluar("Training", yi_train, red.predecir(X_train), clases)
    evaluar("Test", yi_test, red.predecir(X_test), clases)


if __name__ == "__main__":
    main()
