"""Callbacks e laço de treino, compartilhados pelos 3 experimentos."""

from __future__ import annotations

import time

import tensorflow as tf

from . import config


def get_callbacks(model_name: str):
    """Callbacks padrão: checkpoint do melhor modelo, early stopping, LR plateau, TensorBoard.

    O checkpoint e os logs são organizados por `model_name` (ex.: 'model_a_rgb').
    """
    config.ensure_dirs()
    ckpt_path = config.MODELS_DIR / f"{model_name}.keras"
    log_dir = config.LOGS_DIR / model_name

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]


def compile_model(model, learning_rate: float = config.LEARNING_RATE):
    """Compila com Adam + sparse categorical crossentropy (labels inteiros)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, train_ds, val_ds, model_name: str, epochs: int = config.EPOCHS):
    """Treina o modelo e retorna (history, tempo_em_segundos)."""
    callbacks = get_callbacks(model_name)
    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start
    return history, elapsed
