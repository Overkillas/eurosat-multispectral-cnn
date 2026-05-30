"""Avaliação no conjunto de teste: métricas, matriz de confusão e relatórios.

Convenção do projeto: métricas finais são SEMPRE reportadas no test set, nunca
no val (que foi usado para early stopping / seleção do melhor checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config


def _collect_predictions(model, test_ds):
    """Roda o modelo no test set e retorna (y_true, y_pred) como arrays 1D.

    Usa a chamada direta `model(x, training=False)` (eager) em vez de
    `model.predict` num laço — evita retraces e acúmulo de memória batch a batch.
    """
    y_true, y_pred = [], []
    for x, y in test_ds:
        probs = model(x, training=False)
        y_pred.append(np.argmax(probs.numpy(), axis=1))
        y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate_on_test(model, test_ds, class_names=config.CLASS_NAMES) -> dict:
    """Avalia no test set. Retorna dict com accuracy, F1 macro, F1 por classe e predições."""
    y_true, y_pred = _collect_predictions(model, test_ds)
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_per_class": {c: float(v) for c, v in zip(class_names, per_class_f1)},
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def compute_confusion_matrix(y_true, y_pred, class_names=config.CLASS_NAMES):
    """Retorna (matriz np.ndarray, DataFrame rotulado) — contagem absoluta."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    return cm, df


def get_classification_report(y_true, y_pred, class_names=config.CLASS_NAMES) -> str:
    """Relatório textual do sklearn (precision/recall/F1 por classe)."""
    return classification_report(
        y_true, y_pred, target_names=class_names, labels=range(len(class_names)), digits=4
    )


def save_metrics(metrics: dict, path: Path) -> None:
    """Serializa o dict de métricas em JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2))


def load_metrics(path: Path) -> dict:
    """Carrega métricas salvas (usado no notebook de apresentação)."""
    return json.loads(Path(path).read_text())
