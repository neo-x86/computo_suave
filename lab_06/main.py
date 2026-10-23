#!/usr/bin/env python3
"""
Laboratorio 6 - Cómputo Suave
Problema de las 8 reinas con un Algoritmo Genético.

  - Cromosoma octal: 8 genes (0-7). Posición = columna, valor = fila.
  - Selección por torneo con 3 participantes
  - Fitness f(x) = 28 - C(x)

Ejecutar desde esta carpeta:  python main.py
"""

import random

from GAoctal import GA
from interfaz import (
    AptitudReinas,
    MutacionOctal,
    ReproduccionUnCruce,
    SeleccionTorneo,
)
from interfaz.aptitud import contar_conflictos
from utilidades import graficar, imprimir_tablero


def main(semilla: int = 42):
    random.seed(semilla)

    num_genes = 8
    cant_ind = 200
    max_generaciones = 1000
    max_pares = num_genes * (num_genes - 1) // 2  # 28

    variables = {"max_pares": max_pares}

    gaparams = {
        "Aptitud": AptitudReinas(),
        "Seleccion": SeleccionTorneo(3),  # torneo de 3 participantes
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionOctal(0.15),
        "por_remplazo": 0.05,  # 5% de élite
    }

    print("--- PARÁMETROS ---")
    print(f"Población: {cant_ind} | Genes: {num_genes} (octales) | Torneo: 3")
    print(f"Mutación: 0.15 por gen | Élite: 5% | Máx. generaciones: {max_generaciones}\n")

    print("--- EVOLUCIÓN ---")
    ga = GA(cant_ind, num_genes, gaparams)
    gen_final = ga.buscar_solucion(max_generaciones, max_pares, variables, cada=5)

    ga.evaluar.evaluar_poblacion(ga.poblacion, variables)
    ga.poblacion.sort(key=lambda c: c.score, reverse=True)

    print("\n--- RESULTADO ---")
    mejor = ga.poblacion[0]
    print(f"Generaciones: {gen_final} | Mejor fitness: {mejor.score}/{max_pares}")
    print(f"Cromosoma (octal): {mejor}  ->  filas por columna: {mejor.genes()}")
    print(f"Conflictos C(x): {contar_conflictos(mejor.genes())}\n")
    imprimir_tablero(mejor.genes())

    soluciones = sorted({str(c) for c in ga.poblacion if c.score == max_pares})
    print(f"\nSoluciones distintas en la población final: {len(soluciones)}")
    for s in soluciones:
        print("  ", s)

    graficar(ga.historial_mejor, ga.historial_prom)


if __name__ == "__main__":
    main()
