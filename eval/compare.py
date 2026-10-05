"""Compara dos corridas (antes/después): python eval/compare.py baseline mejora"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "resultados"
M = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

a, b = sys.argv[1], sys.argv[2]
ra = json.load(open(OUT / f"{a}_resumen.json", encoding="utf-8"))
rb = json.load(open(OUT / f"{b}_resumen.json", encoding="utf-8"))

print(f"| Métrica | {a} | {b} | Δ |\n|---|---|---|---|")
for m in M:
    va, vb = ra["global"][m], rb["global"][m]
    print(f"| {m} | {va:.3f} | {vb:.3f} | {vb - va:+.3f} |")

x = range(len(M))
plt.figure(figsize=(8, 4))
plt.bar([i - 0.2 for i in x], [ra["global"][m] for m in M], 0.4, label=a)
plt.bar([i + 0.2 for i in x], [rb["global"][m] for m in M], 0.4, label=b)
plt.xticks(list(x), M, rotation=15); plt.ylim(0, 1.05); plt.ylabel("Puntaje (0-1)")
plt.title("Evaluación Ragas: antes vs. después"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / f"comparacion_{a}_vs_{b}.png", dpi=160)
print("Gráfico:", OUT / f"comparacion_{a}_vs_{b}.png")
