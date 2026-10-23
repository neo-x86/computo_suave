import random
from abc import ABC, abstractmethod
from typing import List

from CromosomaOctal import CromosomaOctal


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(
        self,
        poblacion: List[CromosomaOctal],
        potencial: List[CromosomaOctal],
        max_poblacion: int,
    ) -> None:
        pass


class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes: int):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(self, poblacion, potencial, max_poblacion) -> None:
        while len(poblacion) < max_poblacion:
            hijoA = CromosomaOctal(self.num_genes)
            hijoB = CromosomaOctal(self.num_genes)
            padreA, padreB = random.sample(potencial, 2)
            punto_cruce = random.randint(1, self.num_genes - 1)
            # Se recorren TODOS los genes (1..n_genes inclusive)
            for i in range(1, self.num_genes + 1):
                if i <= punto_cruce:
                    hijoA.set_bits(padreA.get_bits(i), i)
                    hijoB.set_bits(padreB.get_bits(i), i)
                else:
                    hijoA.set_bits(padreB.get_bits(i), i)
                    hijoB.set_bits(padreA.get_bits(i), i)
            poblacion.append(hijoA)
            if len(poblacion) < max_poblacion:
                poblacion.append(hijoB)
