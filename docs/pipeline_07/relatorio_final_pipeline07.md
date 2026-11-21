# Relatório Final de Desenvolvimento, Conclusão e Documentação — Pipeline 07

**Projeto:** Robustez e Generalização de Modelos Fundamentais de Visão e Linguagem em Escoamento Bifásico (CCN)  
**Instituição:** Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN) — Centro de Engenharia Nuclear (CEN)  
**Responsável:** Erikson Vieira Queiroz (`feature-develop-yoerikson`)  
**Arquitetura:** Estágio 1 (`YOLOv8n-seg`) + Estágio 2 (`ResNet-18` Fine-Tuned) + Estágio 3 (`LLaMA-3-8B` via Ollama)  
**Artefatos:** `python_scripts/pipeline_07/pipeline_07.py` | `benchmarks/benchmark_pipeline_07.csv`  

---

## 1. Síntese Executiva dos Resultados ($N_{\text{teste}} = 70$ Amostras Cegas)
* **Desempenho Diagnóstico (Estágio 2 — ResNet-18):** Acurácia global de **95,71%** (67 acertos / 3 erros: TP = 33, TN = 34, FP = 1 [ID 33], FN = 2 [IDs 40 e 60]), Precisão de **97,06%** (Incubação), Recall de **94,29%** (Incubação), Especificidade de **97,14%** (Expulsão), F1-Score de **95,65%** e **MCC = 0,9147** (empate técnico exato com o Pipeline 02).
* **Coerência do Motor Textual (Estágio 3 — LLaMA-3-8B):** **100% de aderência lógica** (34 pareceres *Nominal Operation* e 36 pareceres *Critical Interlock*), sem alucinações ou contradições em relação à saída do classificador visual.
* **Desempenho Temporal vs. Pipeline 02:** Latência média reduzida de **17.188,60 ms** para **6.603,11 ms** (**-61,6%**, ganho de **2,60x**; mediana de **6.110,73 ms**), sendo **40,63 ms** (0,62%) no Estágio 1, **38,25 ms** (0,58%) no Estágio 2 e **6.524,12 ms** (98,80%) no Estágio 3.

## 2. Diagnóstico de Gargalos Empíricos
1. **Subdetecção Severa no Estágio 1 (*Domain Shift*):** O `YOLOv8n-seg` com pesos pré-treinados COCO registrou `0` bolhas em **69 de 70 frames (98,57% de omissão)**, detectando apenas 1 instância na amostra `ID 15` ($\alpha_{\text{planar}} = 28,05\%$).
2. **Pressão de Memória RAM em Borda:** A utilização média de RAM atingiu **92,21%** (pico de **94,40%** nos 15,6 GB disponíveis), com CPU ARMv8 média em **57,02%**, refletindo o custo de co-hospedar dois modelos PyTorch e o servidor Ollama 8B.

## 3. Conclusões e Próximos Passos
* **Viabilidade do Desacoplamento Textual:** Substituir o tensor visual por telemetria numérica estruturada no Estágio 3 eliminou a degeneração lexical observada no Moondream2 (Pipeline 02) e acelerou a inferência em 2,60x.
* **Fine-Tuning Necessário:** Anotar ~150 a 300 frames do CCN com máscaras de bolhas/slugs para treinar pesos dedicados (`yolov8n-seg-bubbles.pt`).
* **Otimização de Memória:** Avaliar modelos textuais menores (`LLaMA-3.2-3B` ou `Phi-3-mini`) ou GPU dedicada para reduzir o consumo de RAM abaixo de 75%.
