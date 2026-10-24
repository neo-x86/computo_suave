#!/usr/bin/env python3
"""
Laboratorio 6 - Cómputo Suave: Solución de problemas con algoritmos genéticos

  Ejercicio 1 (reinas):        8 reinas
      - Cromosoma octal: 8 genes (0-7). Posición = columna, valor = fila
      - Selección por torneo con 3 participantes
      - Fitness f(x) = 28 - C(x)

  Ejercicio 2 (procesadores):  balanceo de carga de 6 tareas en 3 procesadores
      - Cromosoma decimal: 6 genes (1-3). Posición = tarea, valor = procesador
      - Selección por ruleta
      - Fitness f(x) = 1 / (1 + M(x)), M(x) = máxima carga entre procesadores

Uso (desde esta carpeta):
    python main.py                  # corre ambos ejercicios
    python main.py reinas           # solo ejercicio 1
    python main.py procesadores     # solo ejercicio 2
    python main.py todos --semilla 7
"""

import argparse
import itertools
import random

from CromosomaDecimal import CromosomaDecimal
from CromosomaOctal import CromosomaOctal
from GA import GA
from interfaz import (
    AptitudProcesadores,
    AptitudReinas,
    MutacionDecimal,
    MutacionOctal,
    ReproduccionUnCruce,
    SeleccionRuleta,
    SeleccionTorneo,
)
from interfaz.aptitud import calcular_cargas, contar_conflictos
from utilidades import graficar, imprimir_asignacion, imprimir_tablero

SEMILLA = 42


# ----------------------------------------------------------------------------
# Ejercicio 1: 8 reinas
# ----------------------------------------------------------------------------
def ejercicio_reinas(semilla: int = SEMILLA):
    random.seed(semilla)

    num_genes = 8
    cant_ind = 200
    max_generaciones = 1000
    max_pares = num_genes * (num_genes - 1) // 2  # 28 = C(8, 2)

    variables = {"max_pares": max_pares}

    gaparams = {
        "Cromosoma": CromosomaOctal,
        "Aptitud": AptitudReinas(),
        "Seleccion": SeleccionTorneo(3),  # torneo de 3 participantes
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionOctal(0.15),
        "por_remplazo": 0.05,  # 5% de élite
    }

    print("=" * 70)
    print("EJERCICIO 1: PROBLEMA DE LAS 8 REINAS")
    print("=" * 70)
    print("--- PARÁMETROS ---")
    print(f"Población: {cant_ind} | Genes: {num_genes} (octales) | Torneo: 3")
    print(
        f"Mutación: 0.15 por gen | Élite: 5% | Máx. generaciones: {max_generaciones}\n"
    )

    print("--- EVOLUCIÓN ---")
    ga = GA(cant_ind, num_genes, gaparams)
    gen_final = ga.buscar_solucion(max_generaciones, max_pares, variables, cada=5)

    ga.evaluar.evaluar_poblacion(ga.poblacion, variables)
    ga.poblacion.sort(key=lambda c: c.score, reverse=True)

    print("\n--- RESULTADO ---")
    mejor = ga.poblacion[0]
    print(f"Generaciones: {gen_final} | Mejor fitness: {mejor.score:g}/{max_pares}")
    print(f"Cromosoma (octal): {mejor}  ->  filas por columna: {mejor.genes()}")
    print(f"Conflictos C(x): {contar_conflictos(mejor.genes())}\n")
    imprimir_tablero(mejor.genes())

    soluciones = sorted({str(c) for c in ga.poblacion if c.score == max_pares})
    print(f"\nSoluciones distintas en la población final: {len(soluciones)}")
    for s in soluciones:
        print("  ", s)

    graficar(
        ga.historial_mejor,
        ga.historial_prom,
        archivo="evolucion_reinas.png",
        titulo="Evolución del GA - 8 reinas",
        optimo=max_pares,
        ylabel="Fitness  f(x) = 28 - C(x)",
    )


# ----------------------------------------------------------------------------
# Ejercicio 2: balanceo de carga en procesadores
# ----------------------------------------------------------------------------
TIEMPOS = [8, 5, 7, 3, 6, 4]  # T1..T6
N_PROCESADORES = 3


def fuerza_bruta(tiempos, n_proc):
    """Verificación: con 3^6 = 729 asignaciones se puede enumerar todo.
    Devuelve (M óptimo, cantidad de asignaciones óptimas)."""
    maximos = [
        max(calcular_cargas(list(asig), tiempos, n_proc))
        for asig in itertools.product(range(1, n_proc + 1), repeat=len(tiempos))
    ]
    optimo = min(maximos)
    return optimo, maximos.count(optimo)


def ejercicio_procesadores(semilla: int = SEMILLA):
    random.seed(semilla)

    num_genes = len(TIEMPOS)
    cant_ind = 30
    max_generaciones = 200

    variables = {"tiempos": TIEMPOS, "n_procesadores": N_PROCESADORES}

    print("=" * 70)
    print("EJERCICIO 2: DISTRIBUCIÓN DE TAREAS EN PROCESADORES")
    print("=" * 70)
    print("--- PROBLEMA ---")
    for i, t in enumerate(TIEMPOS, 1):
        print(f"  T{i}: {t}")
    print(f"Procesadores: {N_PROCESADORES} | Tiempo total de trabajo: {sum(TIEMPOS)}")

    # Ejemplo del enunciado: el cromosoma 121323 debe dar cargas 15, 11, 7
    ejemplo = CromosomaDecimal(num_genes)
    ejemplo.bits = 121323
    cargas = calcular_cargas(ejemplo.genes(), TIEMPOS, N_PROCESADORES)
    print(
        f"\nEjemplo del enunciado, cromosoma {ejemplo}: cargas = {cargas}, "
        f"M = {max(cargas)}, f = {1 / (1 + max(cargas)):.5f}"
    )

    # Cota inferior: ningún reparto puede ser mejor que la carga promedio (redondeada hacia arriba)
    cota = -(-sum(TIEMPOS) // N_PROCESADORES)
    objetivo = 1.0 / (1.0 + cota)

    gaparams = {
        "Cromosoma": CromosomaDecimal,
        "Aptitud": AptitudProcesadores(),
        "Seleccion": SeleccionRuleta(),
        "Reproduccion": ReproduccionUnCruce(num_genes),
        "Mutacion": MutacionDecimal(0.1),
        "por_remplazo": 0.1,  # 10% de élite
    }

    print("\n--- PARÁMETROS ---")
    print(
        f"Población: {cant_ind} | Genes: {num_genes} (1-{N_PROCESADORES}) | Selección: ruleta"
    )
    print(f"Mutación: 0.1 por gen | Élite: 10% | Máx. generaciones: {max_generaciones}")
    print(
        f"Fitness objetivo: 1/(1+{cota}) = {objetivo:.5f}  (carga promedio = {cota})\n"
    )

    print("--- EVOLUCIÓN ---")
    ga = GA(cant_ind, num_genes, gaparams)
    gen_final = ga.buscar_solucion(max_generaciones, objetivo, variables, cada=5)

    ga.evaluar.evaluar_poblacion(ga.poblacion, variables)
    ga.poblacion.sort(key=lambda c: c.score, reverse=True)

    print("\n--- RESULTADO ---")
    mejor = ga.poblacion[0]
    M = max(calcular_cargas(mejor.genes(), TIEMPOS, N_PROCESADORES))
    print(f"Generaciones: {gen_final} | Mejor fitness: {mejor.score:.5f} | M(x) = {M}")
    print(f"Cromosoma: {mejor}  ->  procesador por tarea: {mejor.genes()}\n")
    imprimir_asignacion(mejor.genes(), TIEMPOS, N_PROCESADORES)

    soluciones = sorted(
        {str(c) for c in ga.poblacion if abs(c.score - objetivo) < 1e-12}
    )
    print(f"\nSoluciones óptimas distintas en la población final: {len(soluciones)}")
    for s in soluciones:
        print("  ", s)

    opt, cuantas = fuerza_bruta(TIEMPOS, N_PROCESADORES)
    print(
        f"\n--- VERIFICACIÓN (fuerza bruta, {N_PROCESADORES ** num_genes} asignaciones) ---"
    )
    print(
        f"M óptimo real = {opt} | asignaciones óptimas = {cuantas} "
        f"(incluye permutaciones equivalentes de los procesadores)"
    )
    print("El AG alcanzó el óptimo." if M == opt else "El AG NO alcanzó el óptimo.")

    graficar(
        ga.historial_mejor,
        ga.historial_prom,
        archivo="evolucion_procesadores.png",
        titulo="Evolución del GA - balanceo de carga",
        optimo=objetivo,
        ylabel="Fitness  f(x) = 1 / (1 + M(x))",
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
EJERCICIOS = {
    "reinas": ejercicio_reinas,
    "procesadores": ejercicio_procesadores,
}


def main():
    parser = argparse.ArgumentParser(description="Laboratorio 6 - Algoritmos genéticos")
    parser.add_argument(
        "ejercicio",
        nargs="?",
        default="todos",
        choices=["reinas", "procesadores", "todos"],
        help="ejercicio a ejecutar (por defecto: todos)",
    )
    parser.add_argument(
        "--semilla", type=int, default=SEMILLA, help="semilla aleatoria"
    )
    args = parser.parse_args()

    claves = list(EJERCICIOS) if args.ejercicio == "todos" else [args.ejercicio]
    for i, clave in enumerate(claves):
        if i > 0:
            print("\n")
        EJERCICIOS[clave](args.semilla)


if __name__ == "__main__":
    main()
