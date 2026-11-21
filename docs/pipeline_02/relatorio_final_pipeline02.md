# Avaliação Experimental de um Pipeline Híbrido em Cascata (OpenCV + ResNet-18 + Moondream2) para Diagnóstico de Escoamento Bifásico em Circuitos de Circulação Natural

**Autor:** Erikson Vieira Queiroz  
**Instituição / Projeto:** IPEN/CNEN — Análise de Escoamentos Bifásicos por Visão Computacional e IA  
**Repositório / Branch:** `ai-analysis-two-phase-fluids` (`feature-develop-yoerikson`)  

---

## Resumo

Este trabalho apresenta a implementação, o treinamento em regime de escassez de dados ($N_{\text{treino}} = 30$) e a avaliação experimental de um pipeline híbrido sequencial em três estágios (**Pipeline 02**) para diagnóstico automatizado de regimes de escoamento bifásico (*Incubação* vs. *Expulsão*) em circuitos térmicos de circulação natural. A arquitetura acopla: (1) extração determinística de telemetria geométrica via **OpenCV** (detector de bordas *Canny* e contornos externos), (2) classificação visual profunda via **ResNet-18** com *Transfer Learning* e congelamento das camadas *BatchNormalization*, e (3) verificação multimodal e síntese operacional via **Moondream2** (~1,8B parâmetros) executado localmente via **Ollama**. Avaliado sobre um conjunto estratificado de $N_{\text{teste}} = 70$ imagens inéditas em ambiente exclusivamente CPU (`Qualcomm ARMv8 8C/8T`, 15,6 GB RAM), o classificador atingiu **95,71% de acurácia global**, **97,06% de precisão**, **94,29% de sensibilidade (*recall*)**, **$F_1$-Score de 95,65%** e **Coeficiente de Correlação de Matthews ($MCC$) de 0,9147**. A análise de latência revelou uma assimetria crítica de quatro ordens de grandeza: enquanto os Estágios 1 e 2 operaram em **4,42 ms** e **54,55 ms** (médias), respectivamente, o Estágio 3 (VLM) consumiu **17.158,06 ms** (99,66% do tempo total) e apresentou degeneração de decodificação autoregressiva sob prompts densos bilíngues. Conclui-se que a combinação OpenCV + ResNet-18 oferece viabilidade imediata em tempo real ($\approx 17\text{ Hz}$ em CPU), devendo o VLM atuar sob disparo condicional (*gatekeeper*) apenas em amostras limítrofes.

---

## 1. Introdução

O monitoramento em tempo real da transição entre os regimes de **Incubação** (nucleação incipiente de microbolhas dispersas) e **Expulsão** (coalescência em golfadas de vapor, regime *slug/churn* e risco de instabilidade termo-hidráulica) é fundamental para a segurança operacional de circuitos de circulação natural utilizados em sistemas nucleares passivos. O **Pipeline 02** integra telemetria geométrica explícita, classificação convolucional *few-shot* e auditoria multimodal local.

---

## 2. Metodologia e Ambiente Experimental

* **Particionamento:** 100 imagens experimentais (50 *Expulsão*, 50 *Incubação*) divididas estratificadamente (`random_state=42`) em **30% treino ($N=30$)** e **70% teste cego ($N=70$, sendo 35 *Incubação* e 35 *Expulsão*)**.
* **Treinamento ResNet-18:** Fine-tuning apenas de `layer4` + cabeçote denso (`Dropout(0.3) -> Linear(512, 64) -> ReLU -> Linear(64, 1)`), `BatchNorm2d` congeladas em `.eval()`, 12 épocas (`Adam`, $lr=3\times 10^{-4}$, `BCEWithLogitsLoss`).
* **Hardware Aferido (`psutil`):** CPU Qualcomm ARMv8 (64-bit) Family 8 Model 1 Rev 201 (8C/8T), 15,6 GB RAM, execução em CPU (`torch 2.15.0`) + Ollama local (`moondream`).

---

## 3. Resultados Experimentais e Discussão

### 3.1. Dinâmica de Treinamento da ResNet-18 (12 Épocas)

| Época | Tempo (s) | Loss Treino | Acc Treino (%) | Loss Validação | Acc Validação (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | 2,08 | 0,4405 | 86,67% | 0,0104 | 100,00% |
| **02** | 1,50 | 0,3662 | 93,33% | 0,0947 | 95,00% |
| **03** | 1,60 | 0,0180 | 100,00% | 0,0018 | 100,00% |
| **04** | 1,60 | 0,1415 | 96,67% | 0,0014 | 100,00% |
| **05** | 1,53 | 0,0105 | 100,00% | 0,0829 | 95,00% |
| **06** | 1,60 | 0,0254 | 100,00% | 0,0010 | 100,00% |
| **07** | 1,70 | 0,0092 | 100,00% | 0,0041 | 100,00% |
| **08** | 1,50 | 0,0092 | 100,00% | 0,0001 | 100,00% |
| **09** | 1,63 | 0,0001 | 100,00% | 0,0000 | 100,00% |
| **10** | 1,62 | 0,0000 | 100,00% | 0,0000 | 100,00% |
| **11** | 1,73 | 0,0000 | 100,00% | 0,0001 | 100,00% |
| **12** | 2,00 | 0,0001 | 100,00% | 0,0002 | 100,00% |

![Curvas de Treinamento ResNet-18](figura01_pipeline02.jpg)
*Figura 1 — Evolução da função de perda (`BCEWithLogitsLoss`) e da acurácia de treinamento e validação ao longo das 12 épocas.*

---

### 3.2. Desempenho de Classificação no Conjunto Cego ($N=70$)

* **Verdadeiros Positivos ($TP$):** $33$ | **Verdadeiros Negativos ($TN$):** $34$ | **Falsos Positivos ($FP$):** $1$ (ID `33`) | **Falsos Negativos ($FN$):** $2$ (IDs `40`, `60`)
* **Acurácia:** $95,71\%$ | **Precisão:** $97,06\%$ | **Recall:** $94,29\%$ | **$F_1$-Score:** $95,65\%$ | **$MCC$:** $0,9147$

---

### 3.3. Análise de Latência por Estágio e Telemetria de Hardware ($N=70$)

| Métrica Monitorada | Média | Desvio-Padrão | $Q_1$ (25%) | Mediana ($Q_2$) | $Q_3$ (75%) | Mínimo | Máximo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Latência Estágio 1 — OpenCV (ms)** | 4,42 | 5,58 | 2,22 | 3,30 | 5,19 | 0,86 | 46,45 |
| **Latência Estágio 2 — ResNet-18 (ms)** | 54,55 | 22,65 | 43,84 | 50,27 | 57,46 | 39,96 | 215,94 |
| **Latência Estágio 3 — Moondream2 (ms)** | 17.158,06 | 2.588,29 | 15.997,76 | 16.899,10 | 18.727,94 | 1.719,25 | 21.873,46 |
| **Latência Total Pipeline (ms)** | **17.217,16** | **2.591,21** | **16.060,80** | **16.949,83** | **18.784,72** | **1.765,08** | **21.933,94** |
| **Utilização de CPU (%)** | 65,44% | 12,34% | 56,25% | 60,40% | 71,72% | 54,20% | 99,90% |
| **Utilização de RAM (%)** | 93,87% | 3,54% | 91,53% | 94,75% | 96,80% | 86,10% | 98,40% |
| **Contornos Detectados ($N_{\text{bolhas}}$)** | 232,27 | 156,35 | 103,75 | 219,00 | 366,50 | 7,00 | 566,00 |
| **Fração de Vazios Planar ($\alpha_{\text{planar}}$ %)** | 4,39% | 3,36% | 1,26% | 3,88% | 7,32% | 0,03% | 12,07% |

![Matriz de Confusão e Latência](figura02_pipeline02.jpg)
*Figura 2 — Matriz de confusão ($N=70$) e distribuição de latência por estágio em escala logarítmica.*

![Telemetria Hardware e OpenCV](figura03_pipeline02.jpg)
*Figura 3 — Telemetria de utilização de CPU/RAM e dispersão entre contornos Canny e fração de vazios planar.*

---

### 3.4. Auditoria Qualitativa do VLM (`Moondream2`)

Os Estágios 1 e 2 totalizam apenas **58,97 ms** em média ($\approx 17\text{ FPS}$ em CPU), enquanto o Estágio 3 responde por **99,66% do tempo total** (17,16 s) e eleva a ocupação de RAM para **93,87%**. Além disso, sob `num_predict=45` e prompt bilíngue, o `Moondream2` (~1,8B) exibiu degeneração autoregressiva em ~20% das amostras (gerando loops de caracteres tailandeses como `"รามามามา..."`) e eco literal de prompt/abstenção em ~55% das instâncias.

---

## 4. Conclusões e Recomendações

1. **Disparo Condicional (*Gatekeeper*):** Acionar o Estágio 3 apenas quando a probabilidade da ResNet-18 estiver na faixa de incerteza ($0,15 < P < 0,85$), reduzindo a latência média de **17,22 s** para **~60 ms**.
2. **Padronização Monolíngue e VLM de Maior Capacidade:** Utilizar prompts 100% em inglês e modelos na faixa de 4B–7B parâmetros (`Qwen2.5-VL-7B` ou `LLaVA-NeXT`) com `num_predict >= 80`.
3. **Aceleração por GPU/NPU:** Migrar a inferência para GPU CUDA ou NPU dedicada para aliviar a saturação de RAM (93,87%).
