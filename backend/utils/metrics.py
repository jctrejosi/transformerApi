import numpy as np


def RSE(pred, true):
    """
    Error Cuadrático Relativo (Relative Squared Error):
    Mide qué tan buena es la predicción comparada con simplemente usar el promedio de los datos.
    Un valor cercano a 0 es excelente.
    """
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Correlación (Correlation):
    Mide la relación lineal entre las predicciones y los valores reales.
    Indica si el modelo "sigue la tendencia" (si cuando la realidad sube, el modelo también sube).
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # Se agrega una pequeña constante para evitar división por cero si d es 0
    return (u / d).mean(-1)


def MAE(pred, true):
    """
    Error Absoluto Medio (Mean Absolute Error):
    Es el promedio de las diferencias absolutas. 
    Ejemplo: Si el MAE es 2.5 y predices temperatura, significa que fallas, en promedio, por 2.5°C.
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Error Cuadrático Medio (Mean Squared Error):
    Eleva los errores al cuadrado antes de promediarlos. 
    Castiga mucho más los fallos grandes que el MAE.
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Raíz del Error Cuadrático Medio (Root Mean Squared Error):
    Es la raíz del MSE. Tiene la ventaja de que vuelve a estar en las mismas unidades
    que los datos originales (ej. grados Celsius o humedad %).
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Error Porcentual Absoluto Medio (Mean Absolute Percentage Error):
    Mide el error en términos de porcentaje. 
    Ejemplo: Si da 0.10, significa que el modelo falla un 10% en promedio.
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    Error Porcentual Cuadrático Medio (Mean Squared Percentage Error):
    Similar al MAPE pero eleva los errores porcentuales al cuadrado.
    Útil para detectar si el modelo comete errores porcentuales muy grandes en casos aislados.
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    Calculador Global de Métricas:
    Recibe las predicciones y los valores reales y devuelve un resumen de los 
    indicadores más importantes para evaluar el rendimiento del modelo.
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe