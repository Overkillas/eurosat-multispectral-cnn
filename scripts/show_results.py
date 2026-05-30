"""Imprime um resumo das métricas dos 3 modelos salvos em results/metrics/."""

import glob
import json
import os

files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "metrics", "*.json")))
if not files:
    print("Nenhuma métrica encontrada em results/metrics/.")
for f in files:
    d = json.load(open(f))
    name = os.path.basename(f)
    tt = d.get("train_time_s")
    tt = round(tt) if tt else "?"
    print(f"{name:28} acc={d['accuracy']:.4f}  f1_macro={d['f1_macro']:.4f}  "
          f"params={d.get('num_params')}  tempo={tt}s  canais={d.get('channels')}")
