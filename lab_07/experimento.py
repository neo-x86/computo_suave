import copy

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.model_selection import train_test_split

from algoritmo_genetico import AlgoritmoGenetico
from cromosoma import Cromosoma
from red_neuronal import MLP

SEMILLA = 42


def normalizar(X_train, X_other):
    media, escala = X_train.mean(0), X_train.std(0)
    escala[escala == 0] = 1
    return (X_train - media) / escala, (X_other - media) / escala


def ejecutar(nombre, cargar, ruta, args):
    X, y, columnas = cargar(ruta, args.max_muestras)
    tarea = "clasificacion" if nombre == "spotify" else "regresion"
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEMILLA,
        stratify=y if tarea == "clasificacion" else None,
    )
    X_train, X_test = normalizar(X_train, X_test)
    desnormalizar_y = lambda valores: valores
    if tarea == "clasificacion":
        clases = int(y.max() + 1)
        y_train, y_test = np.eye(clases)[y_train], np.eye(clases)[y_test]
    else:
        media, escala = y_train.mean(0), y_train.std(0)
        escala[escala == 0] = 1
        y_train, y_test = (y_train - media) / escala, (y_test - media) / escala
        desnormalizar_y = lambda valores: valores * escala + media
    val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEMILLA,
        stratify=y_train.argmax(1) if tarea == "clasificacion" else None,
    )
    print(f"\n=== {nombre.upper()} | {len(X)} muestras | {len(columnas)} features ===")
    print(f"Features: {', '.join(columnas)}")
    baseline = Cromosoma(2, (64, 32, 32), 0.001)
    ga = AlgoritmoGenetico(
        X_train.shape[1], y_train.shape[1], tarea,
        args.poblacion, args.generaciones, SEMILLA,
    )
    print("Optimizando hiperparámetros...")
    mejor = ga.evolucionar((val[0], val[1]), (val[2], val[3]), args.epocas_ga)
    mejor = copy.deepcopy(mejor)
    print(
        f"Mejor arquitectura: "
        f"{mejor.arquitectura(X.shape[1], y_train.shape[1])} | "
        f"lr={mejor.lr:.6g}"
    )
    for etiqueta, cromosoma, semilla in (
        ("Baseline", baseline, 7), ("Optimizado", mejor, 8)
    ):
        red = MLP(
            cromosoma.arquitectura(X_train.shape[1], y_train.shape[1]),
            tarea, semilla,
        )
        historial = red.entrenar(
            X_train, y_train, args.epocas, cromosoma.lr, 64,
            np.random.default_rng(semilla),
        )
        pred = red.predecir(X_test)
        if tarea == "clasificacion":
            real, estimada = y_test.argmax(1), pred.argmax(1)
            print(
                f"{etiqueta:10s}: loss={historial[-1]:.6f} | "
                f"accuracy={accuracy_score(real, estimada):.4f} | "
                f"F1={f1_score(real, estimada, average='macro', zero_division=0):.4f}"
            )
        else:
            mse = mean_squared_error(
                desnormalizar_y(y_test), desnormalizar_y(pred)
            )
            print(
                f"{etiqueta:10s}: loss={historial[-1]:.6f} | "
                f"MSE={mse:.6f} | RMSE={np.sqrt(mse):.6f}"
            )
