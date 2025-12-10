"""
Métricas de clasificación para evaluación de modelos.

Este módulo proporciona funciones para calcular métricas estándar de clasificación
binaria incluyendo AUC, accuracy, precision, sensitivity, specificity y F1-score.
"""

import numpy as np
from typing import Dict, Union, Optional
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def calculate_auc(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    average: str = 'macro'
) -> float:
    """
    Calcula el Area Under the ROC Curve (AUC-ROC).

    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred_proba: Probabilidades predichas (valores entre 0 y 1)
        average: Tipo de promedio para multi-clase ('macro', 'weighted', 'micro')

    Returns:
        Valor AUC entre 0 y 1

    Raises:
        ValueError: Si las etiquetas son inválidas o no hay ambas clases
    """
    try:
        # Verificar que haya al menos dos clases diferentes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            raise ValueError(f"Se necesitan al menos 2 clases, solo se encontró: {unique_labels}")

        return float(roc_auc_score(y_true, y_pred_proba, average=average))
    except Exception as e:
        raise ValueError(f"Error calculando AUC: {str(e)}")


def calculate_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calcula la accuracy (exactitud) de las predicciones.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas (0 o 1)

    Returns:
        Accuracy entre 0 y 1
    """
    return float(accuracy_score(y_true, y_pred))


def calculate_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0
) -> float:
    """
    Calcula la precision (valor predictivo positivo).

    Precision = TP / (TP + FP)

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas (0 o 1)
        zero_division: Valor a retornar cuando no hay positivos predichos

    Returns:
        Precision entre 0 y 1
    """
    return float(precision_score(y_true, y_pred, zero_division=zero_division))


def calculate_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0
) -> float:
    """
    Calcula la sensitivity (recall, true positive rate).

    Sensitivity = Recall = TPR = TP / (TP + FN)

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas (0 o 1)
        zero_division: Valor a retornar cuando no hay positivos verdaderos

    Returns:
        Sensitivity entre 0 y 1
    """
    return float(recall_score(y_true, y_pred, zero_division=zero_division))


def calculate_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Calcula la specificity (true negative rate).

    Specificity = TNR = TN / (TN + FP)

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas (0 o 1)

    Returns:
        Specificity entre 0 y 1

    Raises:
        ValueError: Si no hay negativos en y_true
    """
    # Calcular matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Evitar división por cero
    if (tn + fp) == 0:
        raise ValueError("No hay casos negativos en y_true, no se puede calcular specificity")

    return float(tn / (tn + fp))


def calculate_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_division: float = 0.0
) -> float:
    """
    Calcula el F1-score (media armónica de precision y recall).

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas (0 o 1)
        zero_division: Valor a retornar cuando precision + recall = 0

    Returns:
        F1-score entre 0 y 1
    """
    return float(f1_score(y_true, y_pred, zero_division=zero_division))


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    zero_division: float = 0.0
) -> Dict[str, float]:
    """
    Calcula todas las métricas de clasificación en un solo llamado.

    Args:
        y_true: Etiquetas verdaderas (0 o 1)
        y_pred: Etiquetas predichas (0 o 1)
        y_pred_proba: Probabilidades predichas (opcional, necesario para AUC)
        zero_division: Valor para métricas cuando hay división por cero

    Returns:
        Diccionario con todas las métricas:
            - accuracy: Exactitud
            - precision: Precisión
            - sensitivity: Sensibilidad (recall)
            - specificity: Especificidad
            - f1_score: F1-score
            - auc: AUC-ROC (solo si se proporciona y_pred_proba)

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> y_proba = np.array([0.1, 0.9, 0.4, 0.2, 0.8])
        >>> metrics = calculate_all_metrics(y_true, y_pred, y_proba)
        >>> print(metrics)
        {'accuracy': 0.8, 'precision': 1.0, 'sensitivity': 0.667,
         'specificity': 1.0, 'f1_score': 0.8, 'auc': 0.917}
    """
    metrics = {}

    # Métricas básicas (siempre calculadas)
    metrics['accuracy'] = calculate_accuracy(y_true, y_pred)
    metrics['precision'] = calculate_precision(y_true, y_pred, zero_division=zero_division)
    metrics['sensitivity'] = calculate_sensitivity(y_true, y_pred, zero_division=zero_division)
    metrics['f1_score'] = calculate_f1_score(y_true, y_pred, zero_division=zero_division)

    # Specificity (puede fallar si no hay negativos)
    try:
        metrics['specificity'] = calculate_specificity(y_true, y_pred)
    except ValueError as e:
        metrics['specificity'] = float('nan')
        print(f"Warning: {str(e)}")

    # AUC (solo si se proporcionan probabilidades)
    if y_pred_proba is not None:
        try:
            metrics['auc'] = calculate_auc(y_true, y_pred_proba)
        except ValueError as e:
            metrics['auc'] = float('nan')
            print(f"Warning: No se pudo calcular AUC: {str(e)}")

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Métricas de Clasificación") -> None:
    """
    Imprime las métricas de forma formateada.

    Args:
        metrics: Diccionario con las métricas
        title: Título a mostrar
    """
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    for metric_name, metric_value in metrics.items():
        # Formatear nombre de métrica
        display_name = metric_name.replace('_', ' ').title()

        # Manejar valores NaN
        if np.isnan(metric_value):
            value_str = "N/A"
        else:
            value_str = f"{metric_value:.4f}"

        print(f"{display_name:.<30} {value_str:>10}")

    print(f"{'='*50}\n")


def get_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, Union[int, float]]:
    """
    Calcula métricas basadas en la matriz de confusión.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas

    Returns:
        Diccionario con:
            - tp: True Positives
            - tn: True Negatives
            - fp: False Positives
            - fn: False Negatives
            - total: Total de muestras
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total': int(tp + tn + fp + fn)
    }
