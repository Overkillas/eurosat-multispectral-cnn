"""Exporta amostras reais do EuroSAT para testar os modelos.

Por padrão, exporta **uma amostra de cada classe** (10 no total) para a pasta
`sample_images/` na raiz do projeto. Para cada amostra salva:
  - um .npy com as 13 bandas (entrada válida para qualquer modelo);
  - um .png (composição RGB) só para visualizar.

O nome do arquivo inclui a classe verdadeira, então você confere se a predição bate.

Uso:
  python scripts/export_sample.py                 # uma de cada classe -> sample_images/
  python scripts/export_sample.py 20              # 20 amostras (as primeiras do teste)
  python scripts/export_sample.py 0 minha_pasta   # uma por classe, em outra pasta
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import config, data_loader as dl  # noqa: E402
from src import visualization as viz  # noqa: E402

# Argumentos: nº de amostras (0 = uma por classe) e pasta de saída.
n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out_dir = config.PROJECT_ROOT / (sys.argv[2] if len(sys.argv) > 2 else "sample_images")
out_dir.mkdir(parents=True, exist_ok=True)

_, _, test_ds = dl.make_splits()


def save(arr, idx, cls):
    base = out_dir / f"sample_{idx:02d}_{cls}"
    np.save(f"{base}.npy", arr)
    plt.imsave(f"{base}.png", viz.rgb_composite(arr))
    print(f"salvo: {base.name}.npy / .png  (classe real: {cls})")


if n <= 0:
    # Uma amostra de cada classe (cobre as 10 classes).
    seen = {}
    for img, lbl in test_ds:
        c = int(lbl)
        if c not in seen:
            seen[c] = img.numpy()
        if len(seen) == config.NUM_CLASSES:
            break
    for idx, c in enumerate(sorted(seen)):
        save(seen[c], idx, config.CLASS_NAMES[c])
else:
    for i, (img, lbl) in enumerate(test_ds.take(n)):
        save(img.numpy(), i, config.CLASS_NAMES[int(lbl)])

print(f"\nAmostras em {out_dir}")
print("Teste com:  python scripts/predict_file.py sample_images/ model_c_multispectral")
