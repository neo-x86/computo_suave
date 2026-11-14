# Laboratorio 7: algoritmo neurogenético

`main.py` implementa la actividad solicitada para:

- **Spotify**: clasificación multiclase de los diez géneros más frecuentes.
- **Steam**: regresión del precio usando las variables numéricas disponibles.

El cromosoma codifica número de capas ocultas, neuronas por capa y *learning
rate*. El fitness es F1 macro para Spotify y el negativo del MSE para Steam.
Se conserva una élite, se usa selección por torneo implícita sobre la mitad
superior, cruce de un punto y mutación.

Desde `lab_07`:

```bash
python main.py --dataset todos
python main.py --dataset spotify --generaciones 10 --poblacion 12
python main.py --dataset steam --steam-csv /ruta/games.csv
```

Los datasets se descargan la primera vez en `lab_07/data/`. Para una prueba
rápida puede reducirse `--max-muestras`, `--epocas-ga` y `--epocas`.
