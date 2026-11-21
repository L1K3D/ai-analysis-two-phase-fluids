# Plano de Compreensão e Arquitetura — Pipeline 02 (OpenCV + ResNet-18 + Moondream2)

**Autor:** Erikson Vieira Queiroz  
**Projeto:** IPEN/CNEN — Análise de Escoamentos Bifásicos por Visão Computacional e IA  
**Branch:** `feature-develop-yoerikson`  

---

## 1. Objetivo e Escopo

O **Pipeline 02** tem como objetivo diagnosticar em cascata os regimes de escoamento bifásico (**Incubação** vs. **Expulsão**) em circuitos térmicos de circulação natural, combinando visão computacional determinística, classificação profunda sob escassez de dados rotulados ($N_{\text{treino}} = 30$) e auditoria multimodal local sem dependência de nuvem.

## 2. Arquitetura em Três Estágios

1. **Estágio 1 — Extração Geométrica Determinística (OpenCV):**
   * Conversão RGB $\rightarrow$ Escala de Cinza (`cv2.cvtColor`), detecção de bordas via operador de **Canny** (limiares $50$ e $150$) e segmentação de contornos externos (`cv2.findContours`).
   * **Saídas:** Contagem de estruturas gasosas ($N_{\text{bolhas}}$) e fração de vazios planar ($\alpha_{\text{planar}}$).
2. **Estágio 2 — Classificador Profundo (ResNet-18 Transfer Learning):**
   * Backbone pré-treinado na ImageNet-1K com congelamento de `conv1` até `layer3` e ajuste fino apenas de `layer4` e do cabeçote denso (`Dropout(0.3) -> Linear(512, 64) -> ReLU -> Linear(64, 1)`).
   * Camadas `BatchNorm2d` travadas em `.eval()` durante o treinamento de 12 épocas (`Adam`, $lr = 3\times 10^{-4}$) para preservar estatísticas móveis no regime *few-shot* ($N=30$).
3. **Estágio 3 — Auditoria Multimodal e Comando Operacional (Moondream2 via Ollama):**
   * Submissão do frame codificado em JPEG Base64 junto com a probabilidade do classificador e a telemetria do OpenCV ao modelo **Moondream2** (~1,8B parâmetros) rodando localmente em `http://localhost:11434/api/generate`.

## 3. Protocolo de Avaliação e Artefatos

* **Particionamento:** 100 imagens experimentais (50 por classe) divididas estratificadamente (`random_state=42`) em **30% treino ($N=30$)** e **70% teste cego ($N=70$)**.
* **Métricas:** Matriz de confusão ($TP, TN, FP, FN$), Acurácia, Precisão, Recall, $F_1$-Score, $MCC$, latência individual por estágio (ms) e consumo de CPU/RAM (`psutil`).
* **Artefatos Gerados:**
  * `python_scripts/pipeline_02/resnet18_tranning.py` e `resnet18_finetuned.pth`
  * `python_scripts/pipeline_02/pipeline_02.py`
  * `benchmarks/benchmark_pipeline_02.csv`
  * `docs/pipeline_02/relatorio_final_pipeline02.md` e figuras de alta resolução.
