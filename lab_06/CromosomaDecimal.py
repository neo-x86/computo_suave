import random
from typing import List


class CromosomaDecimal:
    """Cromosoma de genes enteros (un dígito decimal por gen), guardado en un
    solo entero. Posición del gen = tarea, valor del gen = procesador (1-3).

    Ejemplo: el cromosoma 121323 es el entero 121323:
        gen 1 (T1) = 1, gen 2 (T2) = 2, gen 3 (T3) = 1, ...
    El gen 1 es el dígito más significativo (el de más a la izquierda)."""

    def __init__(self, genes: int = 6, min_value: int = 1, max_value: int = 3):
        self.n_genes: int = genes
        self.score: float = -1.0
        self.bits: int = 0
        self.min_value: int = min_value
        self.max_value: int = max_value

    def _peso(self, gen: int) -> int:
        """Potencia de 10 asociada al gen (gen 1 -> 10^(n-1))."""
        return 10 ** (self.n_genes - gen)

    def llenar_random_data(self) -> None:
        for i in range(self.n_genes):
            self.set_bits(random.randint(self.min_value, self.max_value), i + 1)

    def get_bits(self, gen: int) -> int:
        return (self.bits // self._peso(gen)) % 10

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits += (valor - self.get_bits(gen)) * self._peso(gen)

    def genes(self) -> List[int]:
        return [self.get_bits(i) for i in range(1, self.n_genes + 1)]

    def __str__(self) -> str:
        return "".join(str(g) for g in self.genes())
