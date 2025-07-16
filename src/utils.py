import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """
    Carga un archivo CSV en un DataFrame de pandas.

    Parámetros
    ----------
    path : str
        Ruta al archivo CSV.
    **kwargs
        Argumentos adicionales pasados a pd.read_csv.

    Retorna
    -------
    pd.DataFrame
        DataFrame con los datos cargados.
    """
    return pd.read_csv(path, low_memory=False, **kwargs)


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen de valores faltantes por columna.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame a analizar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con columnas ['variable', 'faltantes', 'pct_faltantes'] ordenado de mayor a menor.
    """
    total = len(df)
    missing = df.isnull().sum().rename_axis('variable').reset_index(name='faltantes')
    missing['pct_faltantes'] = 100 * missing['faltantes'] / total
    return missing.sort_values('pct_faltantes', ascending=False)


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Elimina columnas con más del umbral de valores faltantes.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    threshold : float, opcional
        Fracción máxima de valores faltantes permitida (por defecto 0.5).

    Retorna
    -------
    pd.DataFrame
        DataFrame con las columnas eliminadas.
    """
    missing = summarize_missing(df)
    drop_cols = missing.loc[missing['pct_faltantes'] > (threshold*100), 'variable']
    return df.drop(columns=drop_cols.tolist())


def fill_numeric_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena valores numéricos nulos con la mediana.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas numéricas.

    Retorna
    -------
    pd.DataFrame
        DataFrame con valores numéricos imputados.
    """
    cols = df.select_dtypes(include=['int64','float64']).columns
    imputer = SimpleImputer(strategy='median')
    df[cols] = imputer.fit_transform(df[cols])
    return df


def fill_categorical_unknown(df: pd.DataFrame, fill_value: str = 'Unknown') -> pd.DataFrame:
    """
    Rellena valores categóricos nulos con una etiqueta dada.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas de tipo object.
    fill_value : str, opcional
        Etiqueta para rellenar nulos (por defecto 'Unknown').

    Retorna
    -------
    pd.DataFrame
        DataFrame con valores categóricos imputados.
    """
    cols = df.select_dtypes(include=['object']).columns
    for c in cols:
        df[c] = df[c].fillna(fill_value)
    return df


def create_dummies(df: pd.DataFrame, columns: list, drop_first: bool = True) -> pd.DataFrame:
    """
    Convierte variables categóricas en variables dummy.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    columns : list
        Lista de columnas a transformar.
    drop_first : bool, opcional
        Si True, elimina la primera dummy (por defecto True).

    Retorna
    -------
    pd.DataFrame
        DataFrame con dummies añadidas.
    """
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


def plot_histograms(df: pd.DataFrame, columns: list, bins: int = 30) -> None:
    """
    Genera histogramas de las columnas numéricas especificadas.

    Parámetros
    ----------
    df : pd.DataFrame
    columns : list
        Lista de columnas numéricas a graficar.
    bins : int, opcional
        Número de bins en el histograma (por defecto 30).
    """
    for col in columns:
        plt.figure(figsize=(6,4))
        plt.hist(df[col].dropna(), bins=bins)
        plt.title(f'Distribución de {col}')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.show()
