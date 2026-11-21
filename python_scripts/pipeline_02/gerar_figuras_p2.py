import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

SCRIPT_DIR = Path(__file__).resolve().parent
project_root = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
     if (parent / "benchmarks" / "benchmark_pipeline_02.csv").is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Não foi possível localizar benchmarks/benchmark_pipeline_02.csv")

csv_path = project_root / "benchmarks" / "benchmark_pipeline_02.csv"
docs_dir = project_root / "docs" / "pipeline_02"
docs_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path, encoding="utf-8")

# Normaliza possíveis problemas de encoding na leitura
def norm_label(val):
    s = str(val)
    if "Incuba" in s:
        return 1
    return 0

y_true = df["Real"].apply(norm_label).values
y_pred = df["Predicao_Classificador"].apply(norm_label).values

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
acc = accuracy_score(y_true, y_pred) * 100
prec = precision_score(y_true, y_pred) * 100
rec = recall_score(y_true, y_pred) * 100
f1 = f1_score(y_true, y_pred) * 100
mcc = matthews_corrcoef(y_true, y_pred)

print("=== MÉTRICAS DE CLASSIFICAÇÃO (N=70) ===")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"Acurácia: {acc:.2f}% | Precisão: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | MCC: {mcc:.4f}\n")

print("=== ESTATÍSTICAS DE TELEMETRIA E LATÊNCIA ===")
for col in ["Lat_Estagio1_ms", "Lat_Estagio2_ms", "Lat_Estagio3_ms", "Latencia_ms", "CPU_Uso_%", "RAM_Uso_%", "Qtd_Bolhas", "Vapor_%"]:
    s = df[col]
    q1, q2, q3 = np.percentile(s, [25, 50, 75])
    print(f"{col:18s} | Média: {s.mean():9.2f} | DP: {s.std():8.2f} | Q1: {q1:9.2f} | Q2(Med): {q2:9.2f} | Q3: {q3:9.2f} | Min: {s.min():8.2f} | Max: {s.max():9.2f}")

# ---------------------------------------------------------
# FIGURA 1: Curvas de Treinamento (Loss e Acurácia - 12 Épocas)
# ---------------------------------------------------------
epocas = np.arange(1, 13)
loss_treino = [0.4405, 0.3662, 0.0180, 0.1415, 0.0105, 0.0254, 0.0092, 0.0092, 0.0001, 0.0000, 0.0000, 0.0001]
loss_val    = [0.0104, 0.0947, 0.0018, 0.0014, 0.0829, 0.0010, 0.0041, 0.0001, 0.0000, 0.0000, 0.0001, 0.0002]
acc_treino  = [86.67, 93.33, 100.00, 96.67, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]
acc_val     = [100.00, 95.00, 100.00, 100.00, 95.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 100.00]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
ax1.plot(epocas, loss_treino, "o-", color="#1f77b4", linewidth=2, label="Loss Treino (N=30)")
ax1.plot(epocas, loss_val, "s--", color="#d62728", linewidth=2, label="Loss Validação (N=20)")
ax1.set_title("Evolução da Perda (BCEWithLogitsLoss)", fontsize=11, fontweight="bold")
ax1.set_xlabel("Época")
ax1.set_ylabel("Loss")
ax1.set_xticks(epocas)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

ax2.plot(epocas, acc_treino, "o-", color="#1f77b4", linewidth=2, label="Acurácia Treino (%)")
ax2.plot(epocas, acc_val, "s--", color="#2ca02c", linewidth=2, label="Acurácia Validação (%)")
ax2.set_title("Evolução da Acurácia por Época", fontsize=11, fontweight="bold")
ax2.set_xlabel("Época")
ax2.set_ylabel("Acurácia (%)")
ax2.set_ylim(80, 102)
ax2.set_xticks(epocas)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig(docs_dir / "figura01_pipeline02.jpg", format="jpg", dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURA 2: Matriz de Confusão e Distribuição de Latência
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

cm = np.array([[tn, fp], [fn, tp]])
im = ax1.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax1.set_title(f"Matriz de Confusão (N=70 | Acc={acc:.2f}%)", fontsize=11, fontweight="bold")
classes = ["Expulsão (0)", "Incubação (1)"]
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(classes)
ax1.set_yticklabels(classes)
ax1.set_xlabel("Predição ResNet-18")
ax1.set_ylabel("Classe Real")

thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{cm[i, j]}\n({cm[i, j]/70*100:.1f}%)",
                 ha="center", va="center", fontsize=11, fontweight="bold",
                 color="white" if cm[i, j] > thresh else "black")

# Boxplot de Latência por Estágio (Escala Logarítmica para comparar ms vs s)
data_lat = [df["Lat_Estagio1_ms"], df["Lat_Estagio2_ms"], df["Lat_Estagio3_ms"], df["Latencia_ms"]]
bp = ax2.boxplot(data_lat, tick_labels=["Estágio 1\n(OpenCV)", "Estágio 2\n(ResNet-18)", "Estágio 3\n(Moondream2)", "Total\nPipeline"],
                 patch_artist=True, showmeans=True)
colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax2.set_yscale("log")
ax2.set_title("Distribuição de Latência por Estágio (Escala Log)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Latência (ms) - Escala Log10")
ax2.grid(True, which="both", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(docs_dir / "figura02_pipeline02.jpg", format="jpg", dpi=300)
plt.close()

# ---------------------------------------------------------
# FIGURA 3: Telemetria de Hardware (CPU e RAM) e Fração de Vazios
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

ids = df["ID_Teste"]
ax1.plot(ids, df["CPU_Uso_%"], "-", color="#ff7f0e", linewidth=1.5, label="Uso de CPU (%)")
ax1.plot(ids, df["RAM_Uso_%"], "--", color="#9467bd", linewidth=1.5, label="Uso de RAM (%)")
ax1.set_title("Telemetria de Hardware ao Longo das 70 Inferências", fontsize=11, fontweight="bold")
ax1.set_xlabel("ID da Imagem de Teste")
ax1.set_ylabel("Utilização (%)")
ax1.set_ylim(40, 105)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend()

inc_mask = y_true == 1
exp_mask = y_true == 0
ax2.scatter(df.loc[exp_mask, "Qtd_Bolhas"], df.loc[exp_mask, "Vapor_%"], alpha=0.75, color="#d62728", label="Expulsão (Real)")
ax2.scatter(df.loc[inc_mask, "Qtd_Bolhas"], df.loc[inc_mask, "Vapor_%"], alpha=0.75, color="#1f77b4", label="Incubação (Real)")
ax2.set_title("Telemetria Geométrica OpenCV: Contornos vs. Vazios", fontsize=11, fontweight="bold")
ax2.set_xlabel("Quantidade de Contornos Detectados (Canny)")
ax2.set_ylabel("Fração de Vazios Planar (%)")
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig(docs_dir / "figura03_pipeline02.jpg", format="jpg", dpi=300)
plt.close()

print(f"\nFiguras salvas com sucesso em: {docs_dir}")
