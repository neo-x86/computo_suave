#!/usr/bin/env python3
import json

W = 5
pesos = [1, 2, 3, 4, 5]  # Los pesos estan de menor a mayor
valores = [5, 2, 4, 5, 1]

resultados = {}

resultado = 0
for i in range(5):
    peso = 0
    valor = 0
    elementos = []
    for j in range(i, 5):
        if peso + pesos[j] > W:
            continue
        elementos.append(j)
        peso += pesos[j]
        valor += valores[j]
    resultado += 1
    resultados[str(resultado)] = {"Peso": peso, "Valor": valor, "Elementos": elementos}

print(json.dumps(resultados, indent=4))
