from typing import List


def imprimir_tablero(filas: List[int]) -> None:
    n = len(filas)
    print("    " + " ".join(f"c{c}" for c in range(n)))
    for r in range(n):
        celdas = " ".join(" Q" if filas[c] == r else " ." for c in range(n))
        print(f"f{r}  {celdas}")


def graficar(
    historial_mejor,
    historial_prom,
    archivo="evolucion.png",
    titulo="Evolución del GA",
    optimo=None,
    ylabel="Fitness",
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib no está instalado; se omite la gráfica)")
        return
    plt.figure(figsize=(7, 4))
    plt.plot(historial_mejor, label="Mejor fitness")
    plt.plot(historial_prom, label="Fitness promedio")
    if optimo is not None:
        plt.axhline(optimo, color="r", linestyle="--", label="Óptimo")
    plt.xlabel("Generación")
    plt.ylabel(ylabel)
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()
    print(f"Gráfica guardada en {archivo}")


def imprimir_asignacion(
    asignacion: List[int], tiempos: List[int], n_procesadores: int
) -> None:
    print(f"{'Procesador':<12}{'Tareas':<24}{'Tiempo':>8}")
    for p in range(1, n_procesadores + 1):
        tareas = [t for t, proc in enumerate(asignacion) if proc == p]
        nombres = ", ".join(f"T{t + 1}" for t in tareas) or "-"
        detalle = "+".join(str(tiempos[t]) for t in tareas) or "0"
        total = sum(tiempos[t] for t in tareas)
        print(f"P{p:<11}{nombres:<24}{total:>8}  ({detalle})")
