# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

from __future__ import annotations

import gzip
import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore


@dataclass(frozen=True)
class RutasProyecto:
    carpeta_entrada: Path = Path("files/input")
    carpeta_salida: Path = Path("files/output")
    carpeta_modelos: Path = Path("files/models")
    ruta_modelo: Path = Path("files/models/model.pkl.gz")
    ruta_metricas: Path = Path("files/output/metrics.json")


COLUMNAS_CATEGORICAS = ("SEX", "EDUCATION", "MARRIAGE")
OBJETIVO_CRUDO = "default payment next month"
OBJETIVO = "default"
COLUMNAS_A_ELIMINAR = ("ID",)


GRILLA_PARAMETROS = {
    "clf__n_estimators": [100, 200, 500],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_split": [2, 5],
    "clf__min_samples_leaf": [1, 2],
}


def leer_csv_desde_zip(ruta_zip: Path) -> pd.DataFrame:
    """Lee el primer CSV encontrado dentro de un archivo .zip."""
    with zipfile.ZipFile(ruta_zip, "r") as zf:
        nombres_csv = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not nombres_csv:
            raise FileNotFoundError(f"No se encontró CSV dentro de: {ruta_zip}")
        with zf.open(nombres_csv[0]) as f:
            return pd.read_csv(f, sep=",", index_col=0)


def cargar_train_y_test(carpeta_entrada: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carga train y test sin depender del orden.
    Busca .zip con 'train' y 'test' en el nombre; si no, usa los dos primeros.
    """
    zips = sorted(carpeta_entrada.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No hay archivos .zip en: {carpeta_entrada}")

    ruta_train = next((p for p in zips if "train" in p.name.lower()), None)
    ruta_test = next((p for p in zips if "test" in p.name.lower()), None)

    if ruta_train is None or ruta_test is None:
        if len(zips) < 2:
            raise FileNotFoundError("Se esperaban al menos 2 zips (train y test).")
        ruta_train, ruta_test = zips[0], zips[1]

    return leer_csv_desde_zip(ruta_train), leer_csv_desde_zip(ruta_test)


def guardar_modelo_gzip(ruta: Path, objeto: Any) -> None:
    """Guarda un objeto pickle comprimido con gzip."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(ruta, "wb") as f:
        pickle.dump(objeto, f)


def escribir_jsonl(ruta: Path, filas: list[dict[str, Any]]) -> None:
    """Escribe una lista de diccionarios como JSON Lines (una línea por dict)."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila) + "\n")



def depurar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paso 1:
    - Renombra 'default payment next month' -> 'default'
    - Elimina ID
    - Elimina registros con EDUCATION==0 o MARRIAGE==0
    - EDUCATION > 4 -> 4 (others)
    - Quita NA
    """
    out = df.copy()

    if OBJETIVO_CRUDO in out.columns:
        out = out.rename(columns={OBJETIVO_CRUDO: OBJETIVO})

    for col in COLUMNAS_A_ELIMINAR:
        if col in out.columns:
            out = out.drop(columns=col)

    if "MARRIAGE" in out.columns:
        out = out.loc[out["MARRIAGE"] != 0]

    if "EDUCATION" in out.columns:
        out = out.loc[out["EDUCATION"] != 0]
        out["EDUCATION"] = out["EDUCATION"].where(out["EDUCATION"] <= 4, 4)

    return out.dropna()


def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa variables explicativas (X) y objetivo (y)."""
    x = df.drop(columns=[OBJETIVO])
    y = df[OBJETIVO]
    return x, y


def crear_modelo_optimizado() -> GridSearchCV:
    """Crea el pipeline + GridSearchCV del RandomForest."""
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(COLUMNAS_CATEGORICAS)),
        ],
        remainder="passthrough",
    )

    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )

    return GridSearchCV(
        estimator=pipe,
        param_grid=GRILLA_PARAMETROS,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )


def calcular_metricas(dataset: str, y_true, y_pred) -> dict[str, Any]:
    """Calcula métricas y retorna diccionario para JSONL."""
    return {
        "type": "metrics",
        "dataset": dataset,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def calcular_matriz_confusion(dataset: str, y_true, y_pred) -> dict[str, Any]:
    """Construye el dict de la matriz de confusión en el formato del enunciado."""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }


def entrenar_evaluar_y_guardar(rutas: RutasProyecto = RutasProyecto()) -> None:
    """Ejecuta todo el flujo: carga, limpia, entrena, evalúa y guarda salidas."""
    train_raw, test_raw = cargar_train_y_test(rutas.carpeta_entrada)

    train = depurar_dataset(train_raw)
    test = depurar_dataset(test_raw)

    x_train, y_train = separar_xy(train)
    x_test, y_test = separar_xy(test)

    modelo = crear_modelo_optimizado()
    modelo.fit(x_train, y_train)

    guardar_modelo_gzip(rutas.ruta_modelo, modelo)

    pred_train = modelo.predict(x_train)
    pred_test = modelo.predict(x_test)

    filas = [
        calcular_metricas("train", y_train, pred_train),
        calcular_metricas("test", y_test, pred_test),
        calcular_matriz_confusion("train", y_train, pred_train),
        calcular_matriz_confusion("test", y_test, pred_test),
    ]

    escribir_jsonl(rutas.ruta_metricas, filas)

if __name__ == "__main__":
    entrenar_evaluar_y_guardar()