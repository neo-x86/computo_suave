#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from activacion import Sigmoid, dSigmoid

SEMILLA = 42


class Capa:
    def __init__(self, entradas: int, neuronas: int, activacion: str = "sigmoid"):
        limite = np.sqrt(6 / (entradas + neuronas))  # Glorot uniforme
        self.pesos = np.random.uniform(-limite, limite, (entradas, neuronas))
        self.bias = np.zeros((1, neuronas))
        self.activacion = activacion  # "sigmoid" o "lineal"
        self.A = None  # Activaciones
        self.X = None  # Entradas

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.X = X
        Z = np.dot(X, self.pesos) + self.bias
        self.A = Sigmoid(Z) if self.activacion == "sigmoid" else Z
        return self.A

    def _derivada(self) -> np.ndarray:
        if self.activacion == "sigmoid":
            return dSigmoid(self.A)
        return np.ones_like(self.A)  # lineal: f'(z) = 1

    def backpropagation(
        self,
        delta_prop: np.ndarray,
        pesos_siguiente: np.ndarray = None,
        lr: float = 0.1,
    ):
        if pesos_siguiente is not None:
            # Capa oculta
            delta = np.dot(delta_prop, pesos_siguiente.T) * self._derivada()
        else:
            # Capa de salida
            delta = delta_prop * self._derivada()

        # Gradientes promediados sobre el batch (lr independiente del tamaño)
        n = self.X.shape[0]
        dW = np.dot(self.X.T, delta) / n
        dB = np.sum(delta, axis=0, keepdims=True) / n

        pesos_viejos = self.pesos.copy()

        # Actualizar parámetros (SGD)
        self.pesos -= lr * dW
        self.bias -= lr * dB

        return delta, pesos_viejos


class RedNeuronal:
    """Perceptrón multicapa: capas ocultas sigmoide + salida configurable."""

    def __init__(self, arquitectura: list, activacion_salida: str = "lineal"):
        # arquitectura = [n_entradas, h1, h2, ..., n_salidas]
        self.capas = []
        for i in range(len(arquitectura) - 1):
            es_salida = i == len(arquitectura) - 2
            self.capas.append(
                Capa(
                    arquitectura[i],
                    arquitectura[i + 1],
                    activacion=activacion_salida if es_salida else "sigmoid",
                )
            )

    def predecir(self, X: np.ndarray) -> np.ndarray:
        A = X
        for capa in self.capas:
            A = capa.feedforward(A)
        return A

    def paso_entrenamiento(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        Y_hat = self.predecir(X)
        loss = 0.5 * np.mean((y - Y_hat) ** 2)

        # Backpropagation, de la salida hacia la entrada
        delta = Y_hat - y
        pesos_sig = None
        for capa in reversed(self.capas):
            delta, pesos_viejos = capa.backpropagation(delta, pesos_sig, lr=lr)
            pesos_sig = pesos_viejos
        return loss

    def entrenar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epocas: int,
        lr: float,
        batch: int,
        cada: int,
    ):
        n = X.shape[0]
        hist_train, hist_test = [], []
        for epoca in range(epocas):
            idx = np.random.permutation(n)
            perdidas = []
            for i in range(0, n, batch):
                b = idx[i : i + batch]
                perdidas.append(self.paso_entrenamiento(X[b], y[b], lr))

            loss_tr = float(np.mean(perdidas))
            loss_val = 0.5 * np.mean((y_val - self.predecir(X_val)) ** 2)
            hist_train.append(loss_tr)
            hist_test.append(loss_val)

            if epoca % cada == 0 or epoca == epocas - 1:
                print(
                    f"Epoch {epoca:4d} | Train loss: {loss_tr:.6f}"
                    f" | Test loss: {loss_val:.6f}"
                )
        return hist_train, hist_test


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def coef_r(y: np.ndarray, y_hat: np.ndarray) -> float:
    if np.std(y) == 0 or np.std(y_hat) == 0:
        return float("nan")
    return float(np.corrcoef(y, y_hat)[0, 1])


def reporte_metricas(nombre, targets, Y, Y_hat):
    ancho = max(12, max(len(t) for t in targets) + 2)
    print(f"\n{nombre}")
    print(f"{'Variable':<{ancho}}{'R^2':>10}{'R':>10}{'RMSE':>14}")
    for j, t in enumerate(targets):
        rmse = np.sqrt(np.mean((Y[:, j] - Y_hat[:, j]) ** 2))
        print(
            f"{t:<{ancho}}{r2_score(Y[:, j], Y_hat[:, j]):>10.4f}"
            f"{coef_r(Y[:, j], Y_hat[:, j]):>10.4f}{rmse:>14.4f}"
        )


def mostrar_ejemplos(y_real, y_pred, n=5):
    print("\n--- TEST RESULTS ---")
    for i in range(n):
        real = ", ".join(f"{v:10.2f}" for v in y_real[i])
        pred = ", ".join(f"{v:10.2f}" for v in y_pred[i])
        print(f"y    : [{real}]\ny_hat: [{pred}]\n")


def graficar(nombre_archivo, targets, hist_train, hist_test, y_test, pred_test):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib no está instalado; se omiten las gráficas)")
        return

    k = len(targets)
    fig, ax = plt.subplots(1, 1 + k, figsize=(4.5 * (1 + k), 4), squeeze=False)
    ax = ax[0]

    ax[0].plot(hist_train, label="Entrenamiento")
    ax[0].plot(hist_test, label="Prueba")
    ax[0].set_xlabel("Época")
    ax[0].set_ylabel("Pérdida (MSE)")
    ax[0].set_title("Curva de aprendizaje")
    ax[0].legend()

    for j, t in enumerate(targets):
        a = ax[j + 1]
        a.scatter(y_test[:, j], pred_test[:, j], s=4, alpha=0.4)
        lim = [
            min(y_test[:, j].min(), pred_test[:, j].min()),
            max(y_test[:, j].max(), pred_test[:, j].max()),
        ]
        a.plot(lim, lim, "r--", label="Ideal")
        a.set_xlabel(f"{t} real")
        a.set_ylabel(f"{t} predicho")
        a.set_title(f"{t} (prueba)")
        a.legend()

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150)
    plt.close(fig)


AIR_FEATURES = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
AIR_TARGETS = ["CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)"]


def cargar_air_quality(ruta):
    # El CSV usa ';' como separador y ',' como decimal
    df = pd.read_csv(ruta, sep=";", decimal=",")
    df = df.dropna(axis=1, how="all").dropna(how="all")

    # Los valores -200 representan mediciones faltantes
    df = df.replace(-200, np.nan)
    df = df[AIR_FEATURES + AIR_TARGETS].dropna()

    return df[AIR_FEATURES].values, df[AIR_TARGETS].values


def cargar_wec(ruta):
    df = pd.read_csv(ruta)
    columnas_x = [f"{c}{i}" for i in range(1, 50) for c in ("X", "Y")]
    X = df[columnas_x].to_numpy(dtype=float)
    y = df[["Total_Power"]].to_numpy(dtype=float)
    return X, y


EXPERIMENTOS = {
    "air": {
        "titulo": "Air Quality (UCI)",
        "descripcion": (
            "Mediciones horarias de un sensor multigás en una ciudad italiana\n"
            "(marzo 2004 - abril 2005)."
        ),
        "ruta": "./data/air_quality/AirQualityUCI.csv",
        "cargar": cargar_air_quality,
        "targets": AIR_TARGETS,
        "ocultas": [16, 8],
        "activacion_salida": "lineal",
        "norm_y": "estandar",  # (y - media) / std
        "epocas": 1000,
        "lr": 0.05,
        "batch": 64,
        "cada": 50,
        "grafica": "resultados_air_quality.png",
    },
    "wec": {
        "titulo": "Wave Energy Farm (Perth, 49 WECs)",
        "descripcion": (
            "Posiciones (X, Y) de 49 convertidores de energía undimotriz\n"
            "y potencia total generada."
        ),
        "ruta": "./data/wave_energy_farm/WEC_Perth_49.csv",
        "cargar": cargar_wec,
        "targets": ["Total_Power"],
        "ocultas": [64, 32],
        "activacion_salida": "sigmoid",
        "norm_y": "minmax",  # a [0, 1] porque la salida es sigmoide
        "epocas": 200,
        "lr": 0.5,
        "batch": 64,
        "cada": 20,
        "grafica": "resultados_wec.png",
    },
}


def normalizar_y(y_train, modo):
    if modo == "minmax":
        ymin, ymax = y_train.min(axis=0), y_train.max(axis=0)
        rango = np.where(ymax > ymin, ymax - ymin, 1.0)
        return (lambda y: (y - ymin) / rango), (lambda z: z * rango + ymin)
    my, sy = y_train.mean(axis=0), y_train.std(axis=0)
    sy = np.where(sy == 0, 1.0, sy)
    return (lambda y: (y - my) / sy), (lambda z: z * sy + my)


def correr(cfg: dict):
    np.random.seed(SEMILLA)

    X, y = cfg["cargar"](cfg["ruta"])
    targets = cfg["targets"]

    print(f"--- DATASET: {cfg['titulo']} ---")
    print(cfg["descripcion"])
    print(f"Muestras válidas tras limpieza: {X.shape[0]}")
    print(f"Entradas (X): {X.shape[1]} variables")
    print("Salidas  (y):", targets)

    # Separación 60% entrenamiento / 40% prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=SEMILLA
    )
    print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Normalización (estadísticos calculados SOLO con entrenamiento)
    mx, sx = X_train.mean(axis=0), X_train.std(axis=0)
    sx[sx == 0] = 1.0
    X_train_n, X_test_n = (X_train - mx) / sx, (X_test - mx) / sx

    norm_y, desnorm_y = normalizar_y(y_train, cfg["norm_y"])
    y_train_n, y_test_n = norm_y(y_train), norm_y(y_test)

    arquitectura = [X.shape[1], *cfg["ocultas"], len(targets)]
    print(
        "Arquitectura:",
        " -> ".join(map(str, arquitectura)),
        f"(ocultas: sigmoide, salida: {cfg['activacion_salida']})",
    )
    print(
        f"Épocas: {cfg['epocas']} | lr: {cfg['lr']}"
        f"| normalización de y: {cfg['norm_y']}"
    )

    red = RedNeuronal(arquitectura, activacion_salida=cfg["activacion_salida"])

    print("\n--- ENTRENAMIENTO ---")
    hist_tr, hist_te = red.entrenar(
        X_train_n,
        y_train_n,
        X_test_n,
        y_test_n,
        epocas=cfg["epocas"],
        lr=cfg["lr"],
        batch=cfg["batch"],
        cada=cfg["cada"],
    )

    # Predicciones en unidades originales
    Y_hat_train = desnorm_y(red.predecir(X_train_n))
    Y_hat_test = desnorm_y(red.predecir(X_test_n))

    print("\n--- GENERAL RESULTS ---")
    reporte_metricas("Training", targets, y_train, Y_hat_train)
    reporte_metricas("Test", targets, y_test, Y_hat_test)

    mostrar_ejemplos(y_test, Y_hat_test)
    graficar(cfg["grafica"], targets, hist_tr, hist_te, y_test, Y_hat_test)


def main():
    _desc = (__doc__ or "").split("\n")
    parser = argparse.ArgumentParser(description=_desc[1] if len(_desc) > 1 else "")

    parser.add_argument(
        "dataset",
        nargs="?",
        default="todos",
        choices=["air", "wec", "todos"],
        help="dataset a ejecutar (por defecto: todos)",
    )
    args = parser.parse_args()

    claves = list(EXPERIMENTOS) if args.dataset == "todos" else [args.dataset]
    for i, clave in enumerate(claves):
        if i > 0:
            print("\n" + "=" * 70 + "\n")
        correr(EXPERIMENTOS[clave])


if __name__ == "__main__":
    main()
