"""Imprime ganhos por classe (C - A) e melhores/piores classes, para a conclusão."""

import json
import os

import numpy as np

base = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")
a = json.load(open(os.path.join(base, "model_a_rgb.json")))
b = json.load(open(os.path.join(base, "model_b_rgb_nir.json")))
c = json.load(open(os.path.join(base, "model_c_multispectral.json")))

classes = list(a["f1_per_class"].keys())
print(f"{'classe':22} {'A':>7} {'B':>7} {'C':>7} {'C-A':>7}")
gains = {}
for cls in classes:
    fa, fb, fc = a["f1_per_class"][cls], b["f1_per_class"][cls], c["f1_per_class"][cls]
    gains[cls] = fc - fa
    print(f"{cls:22} {fa:7.4f} {fb:7.4f} {fc:7.4f} {fc-fa:+7.4f}")

print("\nMaiores ganhos C-A:")
for cls, g in sorted(gains.items(), key=lambda kv: -kv[1])[:3]:
    print(f"  {cls:22} {g:+.4f}")
print("Maiores quedas C-A:")
for cls, g in sorted(gains.items(), key=lambda kv: kv[1])[:3]:
    print(f"  {cls:22} {g:+.4f}")

print("\nResumo geral:")
for name, m in [("A", a), ("B", b), ("C", c)]:
    print(f"  Modelo {name}: acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
          f"tempo={round(m.get('train_time_s', 0))}s params={m.get('num_params')}")
