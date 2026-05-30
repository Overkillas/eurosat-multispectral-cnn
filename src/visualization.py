"""Funções de plotagem reutilizáveis (notebooks de EDA, treino e apresentação)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config


def _save(fig, save_path):
    """Salva a figura em alta resolução, criando a pasta se preciso."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")


def plot_learning_curves(history, save_path=None, title_suffix=""):
    """Loss e accuracy de treino/validação lado a lado.

    Aceita um objeto History do Keras ou um dict (history.history salvo em JSON).
    """
    h = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(h["loss"], label="treino")
    axes[0].plot(h["val_loss"], label="validação")
    axes[0].set_title(f"Loss {title_suffix}".strip())
    axes[0].set_xlabel("época"); axes[0].set_ylabel("loss"); axes[0].legend()

    axes[1].plot(h["accuracy"], label="treino")
    axes[1].plot(h["val_accuracy"], label="validação")
    axes[1].set_title(f"Acurácia {title_suffix}".strip())
    axes[1].set_xlabel("época"); axes[1].set_ylabel("acurácia"); axes[1].legend()

    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_confusion_matrix(cm, class_names=config.CLASS_NAMES, save_path=None, title="Matriz de confusão", normalize=False):
    """Heatmap da matriz de confusão (contagem ou normalizada por linha)."""
    if normalize:
        data = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        data = cm  # mantém inteiros para o formato "d"
        fmt = "d"
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        data, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True,
    )
    ax.set_xlabel("Predito"); ax.set_ylabel("Real"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def _stretch(band):
    """Normaliza uma banda para [0,1] via percentis 2–98 (melhora contraste visual)."""
    lo, hi = np.percentile(band, 2), np.percentile(band, 98)
    return np.clip((band - lo) / (hi - lo + 1e-8), 0, 1)


def plot_band_comparison(image_13bands, save_path=None, band_names=config.BAND_NAMES):
    """Grade 3×5 mostrando a mesma imagem nas 13 bandas (escala de cinza)."""
    img = np.asarray(image_13bands)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for i, ax in enumerate(axes.flat):
        if i < img.shape[-1]:
            ax.imshow(_stretch(img[..., i]), cmap="gray")
            ax.set_title(band_names[i])
        ax.axis("off")
    fig.suptitle("Mesma cena nas 13 bandas Sentinel-2", fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def rgb_composite(image_13bands):
    """Compõe um RGB visível (B4,B3,B2) com stretch para exibição."""
    img = np.asarray(image_13bands)
    r, g, b = (_stretch(img[..., idx]) for idx in config.RGB_INDICES)
    return np.stack([r, g, b], axis=-1)


def plot_class_distribution(labels, class_names=config.CLASS_NAMES, save_path=None):
    """Gráfico de barras da contagem por classe. `labels` é um array de inteiros."""
    counts = np.bincount(np.asarray(labels), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(x=list(class_names), y=counts, ax=ax, color="steelblue")
    ax.set_xlabel("classe"); ax.set_ylabel("nº de imagens")
    ax.set_title("Distribuição de imagens por classe")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def compute_ndvi(image_13bands):
    """NDVI = (B8 - B4) / (B8 + B4). Entrada: imagem (H,W,13) com valores brutos."""
    img = np.asarray(image_13bands, dtype=np.float32)
    nir = img[..., config.NIR_INDEX]
    red = img[..., config.RED_INDEX]
    return (nir - red) / (nir + red + 1e-8)


# --------------------------------------------------------------------------- #
# Comparações entre os 3 modelos (notebook de apresentação)
# --------------------------------------------------------------------------- #
def build_comparison_table(results: dict) -> pd.DataFrame:
    """Tabela comparativa dos modelos.

    `results` é um dict {label_modelo: metrics_dict}, onde metrics_dict contém
    pelo menos accuracy, f1_macro e (opcionalmente) train_time_s e num_params.
    """
    rows = []
    for name, m in results.items():
        rows.append({
            "Modelo": name,
            "Acurácia": round(m.get("accuracy", float("nan")), 4),
            "F1 macro": round(m.get("f1_macro", float("nan")), 4),
            "Tempo treino (s)": round(m["train_time_s"], 1) if "train_time_s" in m else None,
            "Parâmetros": m.get("num_params"),
        })
    return pd.DataFrame(rows)


def plot_curves_side_by_side(histories: dict, save_path=None):
    """Curvas de val_accuracy dos modelos sobrepostas no mesmo eixo."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, h in histories.items():
        hh = h.history if hasattr(h, "history") else h
        ax.plot(hh["val_accuracy"], label=name)
    ax.set_xlabel("época"); ax.set_ylabel("acurácia de validação")
    ax.set_title("Curvas de validação — comparação entre modelos")
    ax.legend()
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_per_class_comparison(results: dict, class_names=config.CLASS_NAMES, save_path=None):
    """Barras agrupadas do F1 por classe para cada modelo."""
    df = pd.DataFrame({name: m["f1_per_class"] for name, m in results.items()}, index=class_names)
    fig, ax = plt.subplots(figsize=(13, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_xlabel("classe"); ax.set_ylabel("F1-score")
    ax.set_title("F1 por classe — quais classes se beneficiam das bandas extras")
    ax.legend(title="modelo")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
