from dataclasses import dataclass
import numpy as np


@dataclass
class Cromosoma:
    capas: int
    neuronas: tuple[int, int, int]
    lr: float
    fitness: float = -np.inf

    @property
    def genes(self):
        return (self.capas, *self.neuronas, self.lr)

    def arquitectura(self, entradas, salidas):
        return [entradas, *self.neuronas[:self.capas], salidas]
