"""Carregamento do EuroSAT e construção dos pipelines tf.data.

A normalização é calculada **uma única vez** sobre as 13 bandas do conjunto de
treino e reusada nos 3 experimentos. Cada modelo apenas seleciona um subconjunto
de canais — assim a comparação entre RGB / RGB+NIR / multi-espectral é justa.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from . import config

AUTOTUNE = tf.data.AUTOTUNE


# --------------------------------------------------------------------------- #
# Carregamento e splits
# --------------------------------------------------------------------------- #
def load_eurosat():
    """Baixa (se necessário) e prepara o EuroSAT all-bands; retorna o builder."""
    builder = tfds.builder(config.DATASET_NAME)
    builder.download_and_prepare()
    return builder


def _to_xy(example):
    """Extrai (imagem float32, label) do dict do tfds (chave 'sentinel2')."""
    image = tf.cast(example[config.IMAGE_KEY], tf.float32)
    label = example[config.LABEL_KEY]
    return image, label


def make_splits(
    seed: int = config.SEED,
    train_frac: float = config.TRAIN_FRAC,
    val_frac: float = config.VAL_FRAC,
    data_dir: str | None = None,
):
    """Retorna (train_ds, val_ds, test_ds) disjuntos e reprodutíveis.

    Usa o fatiamento percentual do tfds, que é determinístico — mesma `seed`
    de embaralhamento posterior garante a mesma ordem entre experimentos.
    Os datasets retornados produzem (imagem(64,64,13) float32, label int).
    """
    p_train = int(round(train_frac * 100))
    p_val = int(round((train_frac + val_frac) * 100))
    split = [
        f"train[:{p_train}%]",
        f"train[{p_train}%:{p_val}%]",
        f"train[{p_val}%:]",
    ]
    train_ds, val_ds, test_ds = tfds.load(
        config.DATASET_NAME,
        split=split,
        shuffle_files=False,
        as_supervised=False,
        data_dir=data_dir,
    )
    train_ds = train_ds.map(_to_xy, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_to_xy, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(_to_xy, num_parallel_calls=AUTOTUNE)
    return train_ds, val_ds, test_ds


# --------------------------------------------------------------------------- #
# Normalização por banda (z-score)
# --------------------------------------------------------------------------- #
def compute_norm_stats(train_ds) -> tuple[np.ndarray, np.ndarray]:
    """Calcula média e desvio-padrão por banda sobre o conjunto de treino.

    Acumula soma e soma dos quadrados por banda em uma única passada para não
    carregar tudo na RAM. Retorna arrays de shape (13,).
    """
    n_bands = config.NUM_BANDS
    count = np.zeros(n_bands, dtype=np.float64)
    total = np.zeros(n_bands, dtype=np.float64)
    total_sq = np.zeros(n_bands, dtype=np.float64)

    for image, _ in train_ds:
        arr = image.numpy().reshape(-1, n_bands).astype(np.float64)
        count += arr.shape[0]
        total += arr.sum(axis=0)
        total_sq += (arr ** 2).sum(axis=0)

    mean = total / count
    var = total_sq / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))  # evita divisão por zero
    return mean.astype(np.float32), std.astype(np.float32)


def save_norm_stats(mean: np.ndarray, std: np.ndarray, path: Path = config.NORM_STATS_PATH) -> None:
    """Salva média/desvio por banda em JSON para reuso nos notebooks de treino."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mean": np.asarray(mean).tolist(), "std": np.asarray(std).tolist()}
    path.write_text(json.dumps(payload, indent=2))


def load_norm_stats(path: Path = config.NORM_STATS_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Carrega média/desvio por banda salvos por `save_norm_stats`."""
    payload = json.loads(Path(path).read_text())
    return (
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# Seleção de canais e normalização (operações tf)
# --------------------------------------------------------------------------- #
def select_channels(image, channel_indices):
    """Seleciona um subconjunto de bandas do tensor (..., 13)."""
    return tf.gather(image, channel_indices, axis=-1)


def normalize_per_band(image, mean, std):
    """Aplica z-score por banda. `mean`/`std` já devem estar fatiados aos canais."""
    mean = tf.constant(np.asarray(mean), dtype=tf.float32)
    std = tf.constant(np.asarray(std), dtype=tf.float32)
    return (image - mean) / std


# --------------------------------------------------------------------------- #
# Data augmentation (camada Keras — flips + rotações de 90°)
# --------------------------------------------------------------------------- #
class RandomRot90(tf.keras.layers.Layer):
    """Rotaciona o batch por um múltiplo aleatório de 90°.

    Rotação de 90° é a única rotação válida para imagens aéreas (não há
    interpolação, então não distorce assinaturas espectrais).
    """

    def __init__(self, seed: int = config.SEED, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def call(self, inputs, training=None):
        if not training:
            return inputs
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32, seed=self.seed)
        return tf.image.rot90(inputs, k=k)


def get_augmentation_layer(seed: int = config.SEED) -> tf.keras.Sequential:
    """Camada de augmentation: flips horizontal/vertical + rotação de 90°.

    Não inclui color jitter nem rotação contínua de propósito — ambos
    alterariam as assinaturas espectrais que o modelo precisa aprender.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            RandomRot90(seed=seed),
        ],
        name="augmentation",
    )


# --------------------------------------------------------------------------- #
# Pipeline completo
# --------------------------------------------------------------------------- #
def build_pipeline(
    ds,
    channel_indices,
    mean,
    std,
    batch_size: int = config.BATCH_SIZE,
    augment: bool = False,
    shuffle: bool = False,
    shuffle_buffer: int = 2048,
    seed: int = config.SEED,
    cache_path=None,
):
    """Monta o pipeline tf.data para um experimento.

    Ordem: select+normalize → [cache] → shuffle → batch → augment → prefetch.
    `mean`/`std` devem ser os arrays completos de 13 bandas; são fatiados aqui
    para os `channel_indices` escolhidos.

    Cache: por padrão **não** faz cache (lê os tfrecords a cada época) — isso
    mantém o uso de RAM baixo, importante em máquinas com pouca memória. Passe
    `cache_path` (um caminho de arquivo único por split) para cachear **em disco**.
    """
    mean_sel = np.asarray(mean)[channel_indices]
    std_sel = np.asarray(std)[channel_indices]

    ds = ds.map(
        lambda x, y: (normalize_per_band(select_channels(x, channel_indices), mean_sel, std_sel), y),
        num_parallel_calls=AUTOTUNE,
    )
    if cache_path is not None:
        # Cache em disco (não em RAM): salva o resultado do map num arquivo.
        from pathlib import Path
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        ds = ds.cache(str(cache_path))
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = get_augmentation_layer(seed=seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
