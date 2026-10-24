import copy
from typing import Dict, List

from CromosomaOctal import CromosomaOctal


class GA:
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        self.poblacion: List[CromosomaOctal] = [CromosomaOctal(num_genes) for _ in range(num_inv)]
        self.num_inv = num_inv
        self.hyperparams = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproduccion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)  # élite que se conserva
        self.historial_mejor: List[int] = []
        self.historial_prom: List[float] = []
        for c in self.poblacion:
            c.llenar_random_data()

    def buscar_solucion(self, max_generaciones: int, objetivo: int, var: Dict, cada: int = 10):
        """Evoluciona hasta alcanzar `objetivo` de fitness o `max_generaciones`.
        Devuelve la generación en la que terminó."""
        generacion = 1
        while generacion <= max_generaciones:
            self.evaluar.evaluar_poblacion(self.poblacion, var)
            self.poblacion.sort(key=lambda c: c.score, reverse=True)

            mejor = self.poblacion[0].score
            promedio = sum(c.score for c in self.poblacion) / self.num_inv
            self.historial_mejor.append(mejor)
            self.historial_prom.append(promedio)

            if generacion == 1 or generacion % cada == 0:
                print(f"[{generacion:04d}] mejor: {mejor:2d} | promedio: {promedio:6.3f}")

            # Heurística de terminación: se encontró un tablero sin conflictos
            if mejor >= objetivo:
                print(f"[{generacion:04d}] mejor: {mejor:2d} | promedio: {promedio:6.3f}  <- solución")
                return generacion

            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)
            # Elitismo: se copian los mejores directo a la nueva generación
            generacion_nueva = copy.deepcopy(self.poblacion[: self.num_remplazo])
            self.reproduccion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)
            # La mutación solo se aplica a los hijos para no destruir la élite
            self.mutacion.mutacion_poblacion(generacion_nueva[self.num_remplazo :])

            self.poblacion = generacion_nueva
            generacion += 1

        return generacion - 1
