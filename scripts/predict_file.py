"""Prediz a classe de uma imagem — ou de todas as imagens de uma pasta.

Formatos aceitos:
  - .npy  -> array (H, W, 13) com as 13 bandas Sentinel-2 (qualquer modelo)
  - .tif  -> imagem Sentinel-2 com 13 bandas (qualquer modelo)
  - .png/.jpg/... -> foto RGB comum (SÓ funciona com o modelo A, e com ressalvas:
                     a imagem precisa ser uma cena de satélite vista de cima para
                     o resultado fazer sentido; fotos fora do domínio dão lixo)

Uso:
  python scripts/predict_file.py minha_imagem.npy model_c_multispectral
  python scripts/predict_file.py foto_satelite.png                 # usa modelo A
  python scripts/predict_file.py my_images/                        # classifica a pasta toda
  python scripts/predict_file.py my_images/ model_a_rgb
"""

import glob
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config, data_loader as dl  # noqa: E402

CHANNELS_BY_MODEL = {
    "model_a_rgb": config.RGB_INDICES,
    "model_b_rgb_nir": config.RGB_NIR_INDICES,
    "model_c_multispectral": config.ALL_INDICES,
}
IMAGE_EXTS = (".npy", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")

# Caches para não recarregar modelo/normalização a cada arquivo.
_models = {}
_norm = None


def load_image(path):
    """Carrega a imagem e retorna (array float32 com bandas no último eixo, n_bandas)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
    elif ext in (".tif", ".tiff"):
        import tifffile
        arr = tifffile.imread(path).astype(np.float32)
    else:  # foto RGB comum
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
        # Foto 0–255 -> aproxima a faixa de reflectância do Sentinel-2 (~0–3000).
        arr = arr * (3000.0 / 255.0)
    if arr.ndim == 2:
        arr = arr[..., None]
    return arr, arr.shape[-1]


def get_model(model_name):
    if model_name not in _models:
        _models[model_name] = tf.keras.models.load_model(config.MODELS_DIR / f"{model_name}.keras")
    return _models[model_name]


def predict_one(path, model_name=None):
    """Retorna (classe_prevista, confianca, top3, modelo_usado) ou levanta ValueError."""
    global _norm
    arr, n_bands = load_image(path)

    name = model_name or ("model_c_multispectral" if n_bands >= 13 else "model_a_rgb")
    channels = CHANNELS_BY_MODEL[name]

    if n_bands < 13 and name != "model_a_rgb":
        raise ValueError(f"{name} precisa de 13 bandas, mas a imagem tem {n_bands}.")
    if n_bands == 3 and name == "model_a_rgb":
        full = np.zeros((*arr.shape[:2], 13), dtype=np.float32)
        full[..., config.RGB_INDICES] = arr  # coloca RGB nas posições B4,B3,B2
        arr = full
    elif n_bands < 13:
        raise ValueError(f"imagem com {n_bands} bandas não suportada (esperado 3 ou 13).")

    arr = tf.image.resize(arr, (config.IMAGE_SIZE, config.IMAGE_SIZE)).numpy()

    if _norm is None:
        _norm = dl.load_norm_stats()
    mean, std = _norm
    x = dl.normalize_per_band(dl.select_channels(arr, channels),
                              np.asarray(mean)[channels], np.asarray(std)[channels])
    probs = get_model(name)(x[None, ...], training=False).numpy()[0]
    order = probs.argsort()[::-1]
    top3 = [(config.CLASS_NAMES[i], float(probs[i])) for i in order[:3]]
    return config.CLASS_NAMES[order[0]], float(probs[order[0]]), top3, name


def collect_files(path):
    """Se for pasta, retorna todos os arquivos de imagem; se for arquivo, retorna [arquivo]."""
    if os.path.isdir(path):
        files = [f for f in sorted(glob.glob(os.path.join(path, "*")))
                 if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
        return files
    return [path]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else None

    files = collect_files(target)
    if not files:
        sys.exit(f"Nenhuma imagem encontrada em '{target}'. Extensões aceitas: {', '.join(IMAGE_EXTS)}")

    # Em modo pasta, se o modelo escolhido exige 13 bandas, ignora as imagens RGB
    # (.png/.jpg) silenciosamente em vez de gerar um erro por arquivo.
    multiband_exts = (".npy", ".tif", ".tiff")
    if os.path.isdir(target) and model_name in ("model_b_rgb_nir", "model_c_multispectral"):
        before = len(files)
        files = [f for f in files if os.path.splitext(f)[1].lower() in multiband_exts]
        skipped = before - len(files)
        if skipped:
            print(f"({skipped} imagem(ns) RGB ignorada(s) — {model_name} exige 13 bandas)")

    if os.path.isdir(target):
        print(f"Classificando {len(files)} imagem(ns) em '{target}':\n")

    results = []
    for f in files:
        try:
            cls, conf, top3, used = predict_one(f, model_name)
        except Exception as e:  # noqa: BLE001
            print(f"  {os.path.basename(f):35} ERRO: {e}")
            continue
        results.append((f, cls, conf))
        if len(files) == 1:
            print(f"\nModelo: {used}  |  imagem: {f}")
            print("Top-3 previsões:")
            for c, p in top3:
                print(f"  {c:22} {p:6.1%}")
            print(f"\n=> Classe prevista: {cls} ({conf:.1%})")
        else:
            print(f"  {os.path.basename(f):35} -> {cls:22} ({conf:.0%})")

    if len(files) > 1:
        print(f"\n{len(results)}/{len(files)} imagens classificadas com sucesso.")


if __name__ == "__main__":
    main()
