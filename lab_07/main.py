#!/usr/bin/env python3
"""Ejecuta el laboratorio 7 de optimización neurogenética."""

import argparse
import random

import numpy as np

from datos import preparar_spotify, preparar_steam
from experimento import SEMILLA, ejecutar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("spotify", "steam", "todos"), default="todos")
    parser.add_argument("--spotify-csv")
    parser.add_argument("--steam-csv")
    parser.add_argument("--max-muestras", type=int, default=12000)
    parser.add_argument("--poblacion", type=int, default=8)
    parser.add_argument("--generaciones", type=int, default=5)
    parser.add_argument("--epocas-ga", type=int, default=5)
    parser.add_argument("--epocas", type=int, default=30)
    args = parser.parse_args()
    np.random.seed(SEMILLA)
    random.seed(SEMILLA)
    if args.dataset in ("spotify", "todos"):
        ejecutar("spotify", preparar_spotify, args.spotify_csv, args)
    if args.dataset in ("steam", "todos"):
        ejecutar("steam", preparar_steam, args.steam_csv, args)


if __name__ == "__main__":
    main()
