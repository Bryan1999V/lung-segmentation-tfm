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


def extract_classification_from_mask(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    method: str = "majority_vote"
) -> tuple:
    """Extract binary classification from multi-class segmentation masks.
    
    For 3-class segmentation (0=background, 1=benign, 2=malignant):
    Converts to binary classification (0=benign, 1=malignant).
    
    Args:
        pred_masks: Predicted segmentation masks (batch_size, H, W) with values [0, 1, 2]
        gt_masks: Ground truth masks (batch_size, H, W) with values [0, 1, 2]
        method: Method to extract classification:
            - "majority_vote": Most common non-background class in nodule region
            - "center_pixel": Class at center of patch
    
    Returns:
        Tuple of (predicted_labels, true_labels, predicted_probabilities)
        - predicted_labels: Binary labels (0: benign, 1: malignant)
        - true_labels: Binary labels (0: benign, 1: malignant)
        - predicted_probabilities: Probabilities for [benign, malignant]
    
    Example:
        >>> pred_masks = np.array([[[0, 0, 1], [1, 1, 0], [0, 0, 0]]])  # benign prediction
        >>> gt_masks = np.array([[[0, 0, 1], [1, 1, 0], [0, 0, 0]]])    # benign ground truth
        >>> pred_labels, true_labels, pred_probas = extract_classification_from_mask(pred_masks, gt_masks)
        >>> print(pred_labels, true_labels)
        [0] [0]  # Both correctly classified as benign
    """
    batch_size = pred_masks.shape[0]
    pred_labels = []
    true_labels = []
    pred_probas = []
    
    for i in range(batch_size):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i]
        
        # Extract true label from ground truth mask
        # Ignore background (0), check if benign (1) or malignant (2)
        nodule_pixels_gt = gt_mask[gt_mask > 0]
        if len(nodule_pixels_gt) > 0:
            # Majority vote in ground truth (should be all same class)
            unique, counts = np.unique(nodule_pixels_gt, return_counts=True)
            gt_class = unique[np.argmax(counts)]  # 1 or 2
            true_label = gt_class - 1  # Convert to 0 (benign) or 1 (malignant)
        else:
            # No nodule in ground truth (shouldn't happen)
            true_label = 0
        
        # Extract predicted label using specified method
        if method == "majority_vote":
            # Look at predicted pixels in nodule region (where gt > 0)
            nodule_region = gt_mask > 0
            if nodule_region.sum() > 0:
                pred_nodule_pixels = pred_mask[nodule_region]
                # Only consider non-background predictions
                pred_nodule_pixels_nonbg = pred_nodule_pixels[pred_nodule_pixels > 0]
                if len(pred_nodule_pixels_nonbg) > 0:
                    unique, counts = np.unique(pred_nodule_pixels_nonbg, return_counts=True)
                    pred_class = unique[np.argmax(counts)]  # 1 or 2
                    pred_label = pred_class - 1
                    # Estimate probability based on pixel counts
                    benign_count = (pred_nodule_pixels == 1).sum()
                    malignant_count = (pred_nodule_pixels == 2).sum()
                    total = benign_count + malignant_count + 1e-7
                    pred_proba = [benign_count / total, malignant_count / total]
                else:
                    # All background predictions -> predict benign by default
                    pred_label = 0
                    pred_proba = [1.0, 0.0]
            else:
                pred_label = 0
                pred_proba = [1.0, 0.0]
        
        elif method == "center_pixel":
            # Use center pixel classification
            h, w = pred_mask.shape
            center_pred = pred_mask[h//2, w//2]
            if center_pred > 0:
                pred_label = center_pred - 1
                pred_proba = [1.0, 0.0] if pred_label == 0 else [0.0, 1.0]
            else:
                pred_label = 0
                pred_proba = [1.0, 0.0]
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        pred_labels.append(pred_label)
        true_labels.append(true_label)
        pred_probas.append(pred_proba)
    
    return np.array(pred_labels), np.array(true_labels), np.array(pred_probas)
