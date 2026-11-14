#!/usr/bin/env python3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def trapmf(x, p):
    """Trapezoidal [a, b, c, d]. Soporta hombros (a==b o c==d)."""
    a, b, c, d = p
    x = np.asarray(x, float)
    y = np.zeros_like(x)
    subida = (x >= a) & (x < b)
    y[subida] = (x[subida] - a) / (b - a)
    y[(x >= b) & (x <= c)] = 1.0
    bajada = (x > c) & (x <= d)
    y[bajada] = (d - x[bajada]) / (d - c)
    return y


def trimf(x, p):
    """Triangular [a, b, c]."""
    a, b, c = p
    return trapmf(x, [a, b, b, c])


def gaussmf(x, p):
    """Gaussiana [sigma, centro]."""
    sigma, c = p
    return np.exp(-((np.asarray(x, float) - c) ** 2) / (2 * sigma**2))


def gbellmf(x, p):
    """Campana generalizada [a, b, c]."""
    a, b, c = p
    return 1.0 / (1.0 + np.abs((np.asarray(x, float) - c) / a) ** (2 * b))


def sigmf(x, p):
    """Sigmoide [pendiente, centro]."""
    a, c = p
    return 1.0 / (1.0 + np.exp(-a * (np.asarray(x, float) - c)))


MF = {
    "trapmf": trapmf,
    "trimf": trimf,
    "gaussmf": gaussmf,
    "gbellmf": gbellmf,
    "sigmf": sigmf,
}


class VariableLinguistica:
    def __init__(self, nombre, rango, puntos=1001):
        self.nombre = nombre
        self.rango = rango
        self.x = np.linspace(rango[0], rango[1], puntos)
        self.terminos = {}  # nombre -> (tipo, params)

    def agregar(self, termino, tipo, params):
        self.terminos[termino] = (tipo, params)
        return self

    def mu(self, termino, x=None):
        tipo, params = self.terminos[termino]
        return MF[tipo](self.x if x is None else x, params)

    def graficar(self, ax):
        for nombre, (tipo, params) in self.terminos.items():
            ax.plot(self.x, self.mu(nombre), label=f"{nombre} ({tipo})")
        ax.set_title(self.nombre)
        ax.set_xlabel(self.nombre)
        ax.set_ylabel("Grado de membresía")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)


# ======================================================================
# Ejercicio 1: tres variables lingüísticas
# ======================================================================
def ejercicio_1():
    print("=" * 70)
    print("EJERCICIO 1: Variables lingüísticas")
    print("=" * 70)

    temperatura = VariableLinguistica("Temperatura (°C)", [0, 40])
    temperatura.agregar("Fría", "trapmf", [0, 0, 8, 16])
    temperatura.agregar("Templada", "gaussmf", [4, 20])
    temperatura.agregar("Cálida", "trimf", [22, 28, 34])
    temperatura.agregar("Caliente", "sigmf", [1.0, 34])

    humedad = VariableLinguistica("Humedad (%)", [0, 100])
    humedad.agregar("Seca", "trapmf", [0, 0, 20, 40])
    humedad.agregar("Confortable", "gbellmf", [15, 3, 50])
    humedad.agregar("Húmeda", "trimf", [55, 75, 95])
    humedad.agregar("Saturada", "sigmf", [0.3, 90])

    velocidad = VariableLinguistica("Velocidad ventilador (%)", [0, 100])
    velocidad.agregar("Apagado", "trapmf", [0, 0, 5, 15])
    velocidad.agregar("Baja", "trimf", [10, 30, 50])
    velocidad.agregar("Media", "gaussmf", [10, 60])
    velocidad.agregar("Alta", "sigmf", [0.2, 85])

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    for v, a in zip([temperatura, humedad, velocidad], ax):
        v.graficar(a)
        print(f"{v.nombre}: {list(v.terminos)}")
    plt.tight_layout()
    plt.savefig("ej1_variables_linguisticas.png", dpi=150)
    plt.close(fig)

    # Fusificación de ejemplo: un valor crisp -> grados de membresía
    t = 25.0
    print(f"\nFusificación de Temperatura = {t}:")
    for nombre in temperatura.terminos:
        print(f"  mu_{nombre}({t}) = {float(temperatura.mu(nombre, np.array([t]))[0]):.4f}")
    print("Gráfica guardada en ej1_variables_linguisticas.png\n")


# ======================================================================
# Ejercicio 2: Unión, Intersección y Complemento (conjuntos tipo 1)
# ======================================================================
def union(mu_a, mu_b, tconorma="max"):
    """A ∪ B. 'max' (estándar) o 'suma_prob' (a + b - a*b, dual del producto)."""
    if tconorma == "max":
        return np.maximum(mu_a, mu_b)
    if tconorma == "suma_prob":
        return mu_a + mu_b - mu_a * mu_b
    raise ValueError("tconorma debe ser 'max' o 'suma_prob'")


def interseccion(mu_a, mu_b, tnorma="min"):
    """A ∩ B. 'min' (estándar) o 'prod' (producto)."""
    if tnorma == "min":
        return np.minimum(mu_a, mu_b)
    if tnorma == "prod":
        return mu_a * mu_b
    raise ValueError("tnorma debe ser 'min' o 'prod'")


def complemento(mu_a):
    """not A = 1 - mu_A."""
    return 1.0 - mu_a


def ejercicio_2():
    print("=" * 70)
    print("EJERCICIO 2: Operadores Unión / Intersección / Complemento")
    print("=" * 70)

    x = np.linspace(0, 1, 1001)
    A = gaussmf(x, [0.1, 0.4])
    B = trimf(x, [0.3, 0.6, 0.9])

    # Comprobación numérica en un caso discreto sencillo
    a = np.array([0.0, 0.3, 0.7, 1.0])
    b = np.array([0.5, 0.5, 0.5, 0.5])
    print("a             =", a)
    print("b             =", b)
    print("union (max)   =", union(a, b))
    print("inters. (min) =", interseccion(a, b))
    print("inters. (prod)=", interseccion(a, b, "prod"))
    print("complemento(a)=", complemento(a))

    fig, ax = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    ax = ax.ravel()
    ax[0].plot(x, A, "b")
    ax[0].set_title("A (gaussmf)")
    ax[1].plot(x, B, "r")
    ax[1].set_title("B (trimf)")
    ax[2].plot(x, complemento(A), "g")
    ax[2].set_title("Complemento de A")
    ax[3].plot(x, union(A, B), "purple")
    ax[3].set_title("A ∪ B (max)")
    ax[4].plot(x, interseccion(A, B), "darkorange", label="min")
    ax[4].plot(x, interseccion(A, B, "prod"), "k--", label="producto")
    ax[4].set_title("A ∩ B")
    ax[4].legend()
    ax[5].plot(x, complemento(union(A, B)), "brown", label="¬(A ∪ B)")
    ax[5].plot(
        x,
        interseccion(complemento(A), complemento(B)),
        "k--",
        label="¬A ∩ ¬B",
    )
    ax[5].set_title("De Morgan")
    ax[5].legend()
    for a_ in ax:
        a_.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plt.savefig("ej2_operadores.png", dpi=150)
    plt.close(fig)

    # De Morgan (verificación)
    dm = np.allclose(complemento(union(A, B)), interseccion(complemento(A), complemento(B)))
    print(f"\nLey de De Morgan ¬(A ∪ B) = ¬A ∩ ¬B se cumple: {dm}")
    print("Gráfica guardada en ej2_operadores.png\n")


# ======================================================================
# Ejercicio 3: composición de relaciones difusas
# ======================================================================
def _tnorma(nombre):
    if nombre == "min":
        return np.minimum
    if nombre == "prod":
        return np.multiply
    raise ValueError("tnorma debe ser 'min' o 'prod'")


def composicion(R, S, tnorma="min"):
    """
    Composición sup-star: T(i,k) = max_j  R(i,j) ⋆ S(j,k)
    tnorma='min'  -> max-min
    tnorma='prod' -> max-producto
    """
    R, S = np.asarray(R, float), np.asarray(S, float)
    if R.shape[1] != S.shape[0]:
        raise ValueError("Dimensiones incompatibles: R es (n,m) y S debe ser (m,p)")
    tn = _tnorma(tnorma)
    return tn(R[:, :, None], S[None, :, :]).max(axis=1)


def ejercicio_3():
    print("=" * 70)
    print("EJERCICIO 3: Composición de relaciones")
    print("=" * 70)

    R = np.array([[1, 0], [0, 1]])
    S = np.array([[1, 0, 0], [1, 0, 1]])
    print("R =\n", R)
    print("S =\n", S)

    T_min = composicion(R, S, "min")
    T_prod = composicion(R, S, "prod")
    print("\nR ∘ S (max-min) =\n", T_min)
    print("R ∘ S (max-producto) =\n", T_prod)

    # Comprobación: R es la identidad, por lo que R ∘ S debe ser igual a S
    print("\nR es la matriz identidad -> R ∘ S debe ser igual a S:",
          np.array_equal(T_min, S) and np.array_equal(T_prod, S))

    # Paso a paso del cálculo de T(1,3) para ilustrar
    print("\nEjemplo T(2,3) = max( min(R21,S13), min(R22,S23) )"
          f" = max( min({R[1,0]},{S[0,2]}), min({R[1,1]},{S[1,2]}) ) = {T_min[1,2]}")

    # Caso no trivial con valores difusos (para comprobar que no es solo identidad)
    R2 = np.array([[0.2, 0.8], [0.6, 0.4]])
    S2 = np.array([[0.5, 0.9, 0.1], [0.7, 0.3, 1.0]])
    print("\nPrueba extra con valores difusos:")
    print("R2 ∘ S2 (max-min) =\n", composicion(R2, S2, "min"))
    print("R2 ∘ S2 (max-prod) =\n", composicion(R2, S2, "prod"))
    print()


# ======================================================================
# Ejercicio 4: Sistema Mamdani (propina) con T-norma configurable
# ======================================================================
class MamdaniFIS:
    def __init__(self, entradas, salida, reglas):
        self.entradas = entradas  # lista de VariableLinguistica
        self.salida = salida
        self.reglas = reglas  # [serv, comida, propina, peso, operador] (1-indexado)

    @staticmethod
    def _tnorma_reducir(valores, metodo):
        r = valores[0]
        for v in valores[1:]:
            r = np.minimum(r, v) if metodo == "min" else r * v
        return r

    def evaluar(self, x, and_m="min", or_m="max", imp_m="min", agg_m="max"):
        """Devuelve (salida crisp, info por regla, mu agregada)."""
        y = self.salida.x
        nombres_sal = list(self.salida.terminos)
        nombres_ent = [list(v.terminos) for v in self.entradas]
        recortes, info = [], []

        for n, regla in enumerate(self.reglas, 1):
            *idx_ent, idx_sal, peso, op = regla
            # 1) Fusificación: grados de membresía de los antecedentes usados
            grados = []
            for k, i in enumerate(idx_ent):
                if i > 0:
                    nombre = nombres_ent[k][i - 1]
                    grados.append(float(self.entradas[k].mu(nombre, np.array([x[k]]))[0]))
            # 2) Fuerza de disparo (AND = T-norma, OR = max)
            if op == 1:  # AND
                alpha = self._tnorma_reducir(grados, and_m)
            else:  # OR
                alpha = max(grados)
            alpha *= peso
            # 3) Implicación (Mamdani min / Larsen producto): alpha ⋆ mu_B(y)
            mu_b = self.salida.mu(nombres_sal[idx_sal - 1])
            res = np.minimum(alpha, mu_b) if imp_m == "min" else alpha * mu_b
            recortes.append(res)
            info.append((n, grados, alpha, res))

        # 4) Agregación
        mu_agg = np.max(recortes, axis=0) if agg_m == "max" else np.sum(recortes, axis=0).clip(0, 1)
        # 5) Defusificación: centroide
        salida = float(np.sum(mu_agg * y) / np.sum(mu_agg)) if mu_agg.sum() > 0 else float("nan")
        return salida, info, mu_agg


def construir_tipper():
    servicio = VariableLinguistica("Servicio", [0, 10])
    servicio.agregar("Malo", "gaussmf", [1.5, 0])
    servicio.agregar("Bueno", "gaussmf", [1.5, 5])
    servicio.agregar("Excelente", "gaussmf", [1.5, 10])

    comida = VariableLinguistica("Comida", [0, 10])
    comida.agregar("Rancia", "trapmf", [0, 0, 1, 3])
    comida.agregar("Deliciosa", "trapmf", [7, 9, 10, 10])

    propina = VariableLinguistica("Propina", [0, 30])
    propina.agregar("Baja", "trimf", [0, 5, 10])
    propina.agregar("Mas o Menos", "trimf", [10, 15, 20])
    propina.agregar("Generosa", "trimf", [20, 25, 30])

    reglas = [
        # servicio, comida, propina, peso, operador (1=AND, 2=OR)
        [1, 1, 1, 1, 2],  # Servicio malo O comida rancia -> propina baja
        [2, 0, 2, 1, 1],  # Servicio bueno -> propina mas o menos
        [3, 2, 3, 1, 2],  # Servicio excelente O comida deliciosa -> generosa
    ]
    return MamdaniFIS([servicio, comida], propina, reglas)


def ejercicio_4():
    print("=" * 70)
    print("EJERCICIO 4: Composición del sistema de propina con T-norma = Producto")
    print("=" * 70)

    fis = construir_tipper()

    # Validación: con mínimo debe reproducir los resultados de la presentación
    print("Validación con T-norma = mínimo (resultados de las diapositivas):")
    for (s, c), esperado in [((1, 9), 15.20), ((10, 3), 24.93), ((5, 5), 15.00)]:
        out, _, _ = fis.evaluar([s, c], and_m="min", imp_m="min")
        print(f"  servicio={s:2d}, comida={c} -> {out:6.2f}%   (presentación: {esperado})")

    # Ejemplo pedido: servicio = 5, comida = 5
    entrada = [5, 5]
    out_min, info_min, agg_min = fis.evaluar(entrada, and_m="min", imp_m="min")
    out_prod, info_prod, agg_prod = fis.evaluar(entrada, and_m="prod", imp_m="prod")

    print(f"\nEntrada: servicio = {entrada[0]}, comida = {entrada[1]}")
    for etiqueta, info, out in [("MÍNIMO", info_min, out_min), ("PRODUCTO", info_prod, out_prod)]:
        print(f"\n  T-norma = {etiqueta} (AND + implicación)")
        for n, grados, alpha, _ in info:
            g = ", ".join(f"{v:.4f}" for v in grados)
            print(f"    Regla {n}: grados antecedente = [{g}] -> fuerza de disparo α = {alpha:.4f}")
        print(f"    Propina (centroide) = {out:.4f}%")

    # Gráfica: implicación y agregación con ambos métodos
    y = fis.salida.x
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), sharey=True)
    for fila, (etiqueta, info, agg, out) in enumerate(
        [("min", info_min, agg_min, out_min), ("producto", info_prod, agg_prod, out_prod)]
    ):
        for n, _, alpha, res in info:
            ax[fila, n - 1].fill_between(y, res, color="green", alpha=0.5)
            ax[fila, n - 1].set_title(f"Regla {n} (α={alpha:.3f}) - impl. {etiqueta}")
        ax[fila, 3].fill_between(y, agg, color="orange", alpha=0.5)
        ax[fila, 3].axvline(out, color="red", label=f"centroide = {out:.2f}")
        ax[fila, 3].set_title(f"Agregación (max) - {etiqueta}")
        ax[fila, 3].legend()
    for a in ax[1]:
        a.set_xlabel("Propina (%)")
    for a in ax[:, 0]:
        a.set_ylabel("Membresía")
    plt.suptitle("Servicio = 5, Comida = 5: mínimo (arriba) vs producto (abajo)")
    plt.tight_layout()
    plt.savefig("ej4_composicion_producto.png", dpi=150)
    plt.close(fig)

    # Comparación en los otros dos ejemplos de la presentación
    print("\nComparación mínimo vs producto en los 3 ejemplos de la presentación:")
    print(f"  {'servicio':>8} {'comida':>7} {'mínimo':>9} {'producto':>10}")
    for s, c in [(1, 9), (10, 3), (5, 5)]:
        o1, _, _ = fis.evaluar([s, c], and_m="min", imp_m="min")
        o2, _, _ = fis.evaluar([s, c], and_m="prod", imp_m="prod")
        print(f"  {s:>8} {c:>7} {o1:>9.4f} {o2:>10.4f}")
    print("\nGráfica guardada en ej4_composicion_producto.png")


if __name__ == "__main__":
    ejercicio_1()
    ejercicio_2()
    ejercicio_3()
    ejercicio_4()
