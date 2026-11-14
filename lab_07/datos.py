from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

SEMILLA = 42
ROOT = Path(__file__).resolve().parent
URL_SPOTIFY = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
URL_STEAM = "https://huggingface.co/datasets/FronkonGames/steam-games-dataset/resolve/main/games.csv"


def cargar_csv(ruta, url, nombre):
    destino = ROOT / "data" / nombre
    if ruta:
        return pd.read_csv(ruta)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        print(f"Descargando {nombre} desde Hugging Face...")
        try:
            urllib.request.urlretrieve(url, destino)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo descargar {nombre}; use un CSV local."
            ) from exc
    return pd.read_csv(destino)


def seleccionar_numericas(df, objetivo, max_features=15):
    columnas = df.drop(columns=[objetivo], errors="ignore").select_dtypes(
        include=np.number
    )
    columnas = columnas.replace([np.inf, -np.inf], np.nan)
    columnas = columnas.loc[:, columnas.notna().any()]
    columnas = columnas.loc[:, columnas.notna().mean() >= 0.5]
    return columnas.iloc[:, :max_features]


def preparar_spotify(ruta, max_muestras):
    df = cargar_csv(ruta, URL_SPOTIFY, "spotify.csv")
    objetivo = "track_genre"
    if objetivo not in df:
        raise ValueError(f"El CSV de Spotify debe contener {objetivo!r}.")
    principales = df[objetivo].value_counts().head(10).index
    df = df[df[objetivo].isin(principales)].copy()
    if len(df) > max_muestras:
        df = df.sample(max_muestras, random_state=SEMILLA)
    features = seleccionar_numericas(df, objetivo)
    datos = pd.concat([features, df[[objetivo]]], axis=1).dropna()
    clases = sorted(datos[objetivo].unique())
    indices = {clase: i for i, clase in enumerate(clases)}
    return (
        datos[features.columns].to_numpy(float),
        datos[objetivo].map(indices).to_numpy(),
        list(features.columns),
    )


def preparar_steam(ruta, max_muestras):
    df = cargar_csv(ruta, URL_STEAM, "steam.csv")
    # El dataset publicado usa nombres con mayúsculas y espacios.
    aliases = {
        columna.strip().casefold(): columna
        for columna in df.columns
    }
    objetivo = aliases.get("price")
    if objetivo is None:
        raise ValueError(
            "El CSV de Steam debe contener una columna 'Price' o 'price'."
        )
    features = seleccionar_numericas(df, objetivo)
    datos = pd.concat([features, df[[objetivo]]], axis=1)
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna()
    datos = datos[datos[objetivo] >= 0]
    if len(datos) > max_muestras:
        datos = datos.sample(max_muestras, random_state=SEMILLA)
    return (
        datos[features.columns].to_numpy(float),
        datos[[objetivo]].to_numpy(float),
        list(features.columns),
    )
