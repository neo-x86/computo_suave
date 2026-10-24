import random
from abc import ABC, abstractmethod
from typing import List

from CromosomaOctal import CromosomaOctal


class Mutacion(ABC):
    @abstractmethod
    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        pass


class MutacionCambioGen(Mutacion):
    """Con probabilidad `prob` por gen, cambia el gen por otro valor distinto
    dentro de su rango [min_value, max_value]. (El flip por XOR del PDF binario
    no aplica a genes octales/decimales.)"""

    def __init__(self, probabilidad: float):
        super().__init__()
        self.prob = probabilidad

    def mutacion_poblacion(self, poblacion: List[CromosomaOctal]):
        for crom in poblacion:
            for i in range(1, crom.n_genes + 1):
                if random.random() < self.prob:
                    actual = crom.get_bits(i)
                    nuevo = random.choice(
                        [
                            v
                            for v in range(crom.min_value, crom.max_value + 1)
                            if v != actual
                        ]
                    )
                    crom.set_bits(nuevo, i)


class MutacionOctal(MutacionCambioGen):
    """Encodificado octal"""


class MutacionDecimal(MutacionCambioGen):
    """Decimal"""
