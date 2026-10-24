import bisect
import random
from abc import ABC, abstractmethod
from typing import List

from CromosomaOctal import CromosomaOctal


class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(
        self, poblacion: List[CromosomaOctal]
    ) -> List[CromosomaOctal]:
        pass


class SeleccionTorneo(Seleccion):
    def __init__(self, participantes: int = 3):
        super().__init__()
        self.participantes = participantes

    def seleccion_poblacion(
        self, poblacion: List[CromosomaOctal]
    ) -> List[CromosomaOctal]:
        ganadores = []
        for _ in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganadores.append(max(competidores, key=lambda c: c.score))
        return ganadores


class SeleccionRuleta(Seleccion):
    """Selección proporcional al fitness (ruleta).

    prob_i = score_i / sum(scores). Se simula la ruleta con las probabilidades
    acumuladas: se "tira" un número aleatorio en [0, 1) y gana el cromosoma
    cuyo tramo acumulado lo contiene. Se tiran tantas veces como individuos."""

    def seleccion_poblacion(
        self, poblacion: List[CromosomaOctal]
    ) -> List[CromosomaOctal]:
        total = sum(c.score for c in poblacion)
        if total <= 0:  # sin información de fitness: selección uniforme
            return [random.choice(poblacion) for _ in poblacion]

        acumuladas = []
        suma = 0.0
        for c in poblacion:
            suma += c.score / total
            acumuladas.append(suma)

        elegidos = []
        for _ in range(len(poblacion)):
            tiro = random.random()
            indice = bisect.bisect_left(acumuladas, tiro)
            elegidos.append(poblacion[min(indice, len(poblacion) - 1)])
        return elegidos
