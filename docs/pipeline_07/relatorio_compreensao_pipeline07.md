# Plano de Compreensão e Arquitetura — Pipeline 07 (YOLOv8n-seg + ResNet-18 + LLaMA-3-8B)

**Autor:** Erikson Vieira Queiroz  
**Instituição:** Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN) — Centro de Engenharia Nuclear (CEN)  
**Ambiente Experimental:** Circuito de Circulação Natural (CCN)  
**Branch:** `feature-develop-yoerikson`  

---

## 1. Objetivo e Escopo Científico

O **Pipeline 07** investiga o equilíbrio entre a precisão da visão computacional profunda (segmentação de instâncias + classificação residual) e a capacidade de raciocínio operacional de um Modelo de Linguagem Denso puramente textual (**LLaMA-3-8B**) no monitoramento do Circuito de Circulação Natural (CCN)[cite: 17]. Diferente dos pipelines baseados em VLMs (como o Pipeline 02 com Moondream2)[cite: 15], o Estágio 3 deste pipeline **não recebe o tensor da imagem**, operando estritamente sobre a abstração matemática e telemétrica extraída nos Estágios 1 e 2[cite: 17].

## 2. Arquitetura Modular em Três Estágios

1. **Estágio 1 — Segmentação de Instâncias (`YOLOv8n-seg`):**
   * Detecção e reconstrução de máscaras binárias de bolhas/vapor via `YOLOv8n-seg` (PyTorch/Ultralytics)[cite: 17].
   * Cálculo da contagem de instâncias ($N_{\text{bolhas}}$) e da fração planar de vazio ($\alpha_{\text{planar}}$) pela união lógica das máscaras binarizadas ($\tau_{\text{mask}} = 0,5$), evitando dupla contagem por sobreposição[cite: 3, 4].
   * **Hipótese de Risco (*Domain Shift*):** Como o modelo utiliza pesos pré-treinados genéricos (COCO) sem ajuste fino em bolhas bifásicas, espera-se subdetecção severa no Estágio 1, permitindo medir como o LLM textual reage a entradas telemétricas esparsas ou nulas[cite: 3, 8].
2. **Estágio 2 — Classificação Visual Profunda (`ResNet-18` Fine-Tuned):**
   * Reutilização exata dos pesos treinados em regime de escassez ($N_{\text{treino}} = 30$) e do conjunto cego ($N_{\text{teste}} = 70$) do Pipeline 02 (`resnet18_finetuned.pth` e `dados_teste_p2.npz`), isolando o efeito da troca do Estágio 1 e do Estágio 3.
3. **Estágio 3 — Motor de Raciocínio Textual (`LLaMA-3-8B` via Ollama):**
   * Submissão de prompt estruturado contendo apenas a classe predita ($\hat{c}$), a confiança posterior ($\%$), a contagem de bolhas ($N_{\text{bolhas}}$) e a fração de vapor ($\alpha_{\text{planar}}$) ao `llama3` (8B quantizado em 4-bit) via API REST local (`localhost:11434/api/generate`)[cite: 17].

## 3. Protocolo de Avaliação e Riscos de Implantação

* **Métricas Registradas (`benchmarks/benchmark_pipeline_07.csv`):** Acurácia, Precisão, Recall, $F_1$-Score, $MCC$, latência individual por estágio (`Lat_Estagio1_ms`, `Lat_Estagio2_ms`, `Lat_Estagio3_ms`), latência total e consumo de CPU/RAM (`psutil`).
* **Risco Principal (Gargalo de Memória RAM/CPU):** Manter simultaneamente em memória o `YOLOv8n-seg`, a `ResNet-18` e o `LLaMA-3-8B` (~4,7 GB em RAM) em CPU ARM64 com 15,6 GB de RAM exige gerenciamento estrito de cache e *warm-up* controlado[cite: 17].
