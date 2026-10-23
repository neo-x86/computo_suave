import random
from abc import ABC, abstractmethod
from typing import List

from CromosomaOctal import CromosomaOctal


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass


class MutacionOctal(Mutacion):
    """Con probabilidad `prob` por gen, cambia el gen por otro valor octal (0-7).
    (El flip por XOR del PDF binario no aplica directo a genes octales,
    por eso se reemplaza el valor por uno distinto dentro de 0-7.)"""

    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    actual = crom.get_bits(i)
                    nuevo = random.choice([v for v in range(crom.max_value + 1) if v != actual])
                    crom.set_bits(nuevo, i)
