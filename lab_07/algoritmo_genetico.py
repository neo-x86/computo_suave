import copy
import random

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error

from cromosoma import Cromosoma
from red_neuronal import MLP


class AlgoritmoGenetico:
    def __init__(self, entradas, salidas, tarea, poblacion, generaciones, semilla):
        self.entradas, self.salidas, self.tarea = entradas, salidas, tarea
        self.generaciones = generaciones
        self.rng = random.Random(semilla)
        self.poblacion = [self.nuevo() for _ in range(poblacion)]
        self.historial = []

    def nuevo(self):
        return Cromosoma(
            self.rng.randint(1, 3),
            tuple(self.rng.choice((16, 32, 64, 96, 128)) for _ in range(3)),
            10 ** self.rng.uniform(-4, -2),
        )

    def evaluar(self, cromosoma, X, y, epocas, semilla):
        red = MLP(
            cromosoma.arquitectura(self.entradas, self.salidas),
            self.tarea,
            semilla,
        )
        red.entrenar(
            X[0], y[0], epocas, cromosoma.lr, 64,
            np.random.default_rng(semilla),
        )
        pred = red.predecir(X[1])
        if self.tarea == "clasificacion":
            score = f1_score(
                y[1].argmax(1), pred.argmax(1),
                average="macro", zero_division=0,
            )
        else:
            score = -float(mean_squared_error(y[1], pred))
        cromosoma.fitness = score

    def evolucionar(self, X, y, epocas):
        for generacion in range(self.generaciones):
            for i, cromosoma in enumerate(self.poblacion):
                self.evaluar(
                    cromosoma, X, y, epocas,
                    1000 + generacion * 100 + i,
                )
            self.poblacion.sort(key=lambda c: c.fitness, reverse=True)
            self.historial.append(self.poblacion[0].fitness)
            print(
                f"  Generación {generacion + 1:02d}: "
                f"fitness={self.poblacion[0].fitness:.6f}"
            )
            if generacion == self.generaciones - 1:
                break
            elite = [
                copy.deepcopy(c)
                for c in self.poblacion[:max(1, len(self.poblacion) // 4)]
            ]
            while len(elite) < len(self.poblacion):
                padres = self.rng.sample(
                    self.poblacion[:max(2, len(self.poblacion) // 2)], 2
                )
                padre, madre = padres
                punto = self.rng.randint(1, 3)
                genes = [
                    padre.neuronas[i] if i < punto else madre.neuronas[i]
                    for i in range(3)
                ]
                hijo = Cromosoma(
                    padre.capas if self.rng.random() < 0.5 else madre.capas,
                    tuple(genes),
                    padre.lr if self.rng.random() < 0.5 else madre.lr,
                )
                if self.rng.random() < 0.2:
                    hijo.capas = self.rng.randint(1, 3)
                if self.rng.random() < 0.3:
                    indice = self.rng.randrange(3)
                    valores = (16, 32, 64, 96, 128)
                    hijo.neuronas = tuple(
                        self.rng.choice(valores) if j == indice else valor
                        for j, valor in enumerate(hijo.neuronas)
                    )
                if self.rng.random() < 0.3:
                    hijo.lr = 10 ** self.rng.uniform(-4, -2)
                elite.append(hijo)
            self.poblacion = elite
        return max(self.poblacion, key=lambda c: c.fitness)
