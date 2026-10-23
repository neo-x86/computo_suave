import random
from typing import Dict, List


class CromosomaOctal:
    """8 genes de 3 bits (valores 0-7) almacenados en un solo entero (24 bits).
    Posición del gen = columna, valor del gen = fila de la reina."""

    def __init__(self, genes: int = 8):
        self.n_genes: int = genes
        self.score: int = -1
        self.gen_width: int = 3  # bits por gen
        self.bits: int = 0
        self.max_value: int = 2**self.gen_width - 1  # 7
        self.mascaras: Dict = {}
        # El gen 1 queda en los bits más significativos (dígito más a la izquierda)
        for g, i in enumerate(reversed(range(0, genes * self.gen_width, self.gen_width))):
            self.mascaras["p" + str(g + 1)] = i
            self.mascaras["m" + str(g + 1)] = self.max_value << i

    def llenar_random_data(self) -> None:
        for i in range(self.n_genes):
            self.set_bits(random.randint(0, self.max_value), i + 1)

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m" + str(gen)]) >> self.mascaras["p" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = (self.bits & ~self.mascaras["m" + str(gen)]) | (
            valor << self.mascaras["p" + str(gen)]
        )

    def genes(self) -> List[int]:
        """Lista de filas: genes()[columna] = fila de la reina."""
        return [self.get_bits(i) for i in range(1, self.n_genes + 1)]

    def __str__(self) -> str:
        return "".join(str(g) for g in self.genes())
