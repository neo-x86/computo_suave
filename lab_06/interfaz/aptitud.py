from abc import ABC, abstractmethod
from typing import Dict, List

from CromosomaOctal import CromosomaOctal


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict):
        """Interfaz encargada de darle un score a la población."""


def contar_conflictos(filas: List[int]) -> int:
    """C(x): pares de reinas que se atacan (misma fila o misma diagonal).
    Como hay una reina por columna, nunca comparten columna."""
    n = len(filas)
    conflictos = 0
    for i in range(n):
        for j in range(i + 1, n):
            misma_fila = filas[i] == filas[j]
            misma_diagonal = abs(filas[i] - filas[j]) == abs(i - j)
            if misma_fila or misma_diagonal:
                conflictos += 1
    return conflictos


class AptitudReinas(Aptitud):
    def evaluar_poblacion(self, poblacion: List[CromosomaOctal], var: Dict):
        max_pares = var["max_pares"]  # 28 = C(8, 2)
        for cromosoma in poblacion:
            # f(x) = 28 - C(x)
            cromosoma.score = max_pares - contar_conflictos(cromosoma.genes())
