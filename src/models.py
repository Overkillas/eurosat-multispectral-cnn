"""Arquitetura da CNN — única para os 3 experimentos.

A *única* diferença entre os modelos A/B/C é `input_shape` (3, 4 ou 13 canais).
Tudo o mais é idêntico, o que isola o efeito das bandas extras.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from . import config


def _conv_block(x, filters: int):
    """Dois Conv2D(3x3) com BN+ReLU seguidos de MaxPool 2x2."""
    for _ in range(2):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D(2)(x)


def build_cnn(input_shape, num_classes: int = config.NUM_CLASSES) -> tf.keras.Model:
    """CNN clássica (~500k params): 3 blocos conv + GAP + Dropout + softmax.

    A camada final é forçada a float32 para manter estabilidade numérica
    quando mixed precision (float16) está ativada na GPU.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = _conv_block(inputs, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, outputs, name="eurosat_cnn")
