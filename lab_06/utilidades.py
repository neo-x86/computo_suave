from typing import List


def imprimir_tablero(filas: List[int]) -> None:
    n = len(filas)
    print("    " + " ".join(f"c{c}" for c in range(n)))
    for r in range(n):
        celdas = " ".join(" Q" if filas[c] == r else " ." for c in range(n))
        print(f"f{r}  {celdas}")


def graficar(historial_mejor, historial_prom, archivo="evolucion_reinas.png"):
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
    plt.axhline(28, color="r", linestyle="--", label="Óptimo (28)")
    plt.xlabel("Generación")
    plt.ylabel("Fitness  f(x) = 28 - C(x)")
    plt.title("Evolución del GA - 8 reinas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()
    print(f"Gráfica guardada en {archivo}")
