"""Sanity check de um modelo treinado.

Recarrega o modelo .keras salvo, re-avalia no conjunto de teste (confirma que a
acurácia bate com a métrica salva) e mostra previsões em imagens individuais.

Uso:
    python scripts/predict_demo.py                      # usa o modelo C (13 bandas)
    python scripts/predict_demo.py model_a_rgb          # ou model_b_rgb_nir
    python scripts/predict_demo.py model_a_rgb 12       # nº de amostras a exibir
"""

import os
import sys

# Garante que o pacote `src` (na raiz do projeto) seja importável.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")  # salva a figura em arquivo (sem janela)
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src import config, data_loader as dl
from src import evaluation, visualization as viz

# Mapeia o nome do modelo para os canais que ele usa.
CHANNELS_BY_MODEL = {
    "model_a_rgb": config.RGB_INDICES,
    "model_b_rgb_nir": config.RGB_NIR_INDICES,
    "model_c_multispectral": config.ALL_INDICES,
}

model_name = sys.argv[1] if len(sys.argv) > 1 else "model_c_multispectral"
n_show = int(sys.argv[2]) if len(sys.argv) > 2 else 10
channels = CHANNELS_BY_MODEL[model_name]

# Carrega o modelo salvo (prova que o .keras persistiu corretamente).
model_path = config.MODELS_DIR / f"{model_name}.keras"
model = tf.keras.models.load_model(model_path)
print(f"Modelo recarregado de {model_path}")

# Reconstrói o pipeline de teste com os mesmos canais e normalização.
_, _, test_ds = dl.make_splits()
mean, std = dl.load_norm_stats()
test_pipe = dl.build_pipeline(test_ds, channels, mean, std)

# 1) Re-avaliação independente — deve bater com a métrica salva.
res = evaluation.evaluate_on_test(model, test_pipe)
print(f"Acurácia (re-avaliada no teste): {res['accuracy']:.4f}  |  F1 macro: {res['f1_macro']:.4f}")

# 2) Inferência em amostras individuais: predito vs real + confiança.
images, labels = next(iter(test_pipe))           # primeiro batch (normalizado)
probs = model(images, training=False).numpy()
preds = probs.argmax(axis=1)
confs = probs.max(axis=1)

# Para exibir, pega as mesmas amostras cruas (RGB com realce) do test set.
raw_imgs = [img.numpy() for img, _ in test_ds.take(n_show)]
raw_lbls = [int(lbl) for _, lbl in test_ds.take(n_show)]

cols = 5
rows = (n_show + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
correct = 0
for i, ax in enumerate(axes.flat):
    if i >= n_show:
        ax.axis("off"); continue
    ax.imshow(viz.rgb_composite(raw_imgs[i]))
    pred, true, conf = preds[i], raw_lbls[i], confs[i]
    ok = pred == true
    correct += ok
    color = "green" if ok else "red"
    ax.set_title(
        f"pred: {config.CLASS_NAMES[pred]} ({conf:.0%})\nreal: {config.CLASS_NAMES[true]}",
        color=color, fontsize=9,
    )
    ax.axis("off")
fig.suptitle(f"{model_name} — {correct}/{n_show} corretos nas amostras exibidas", fontsize=13)
fig.tight_layout()

out = config.FIGURES_DIR / f"{model_name}_demo.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figura de previsões salva em: {out}")
print(f"Acertos nas {n_show} amostras exibidas: {correct}/{n_show}")
