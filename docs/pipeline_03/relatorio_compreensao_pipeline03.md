# Relatório Técnico de Compreensão e Viabilidade: Pipeline 03
**Projeto de Doutorado:** Robustez e Generalização de Modelos Fundamentais de Visão e Linguagem em Escoamento Bifásico  
**Instituição:** Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN)  
**Ambiente Experimental:** Circuito de Circulação Natural (CCN)  
**Pipeline Analisado:** Pipeline 03 (OpenCV + EfficientNet-B0 + LLaMA-3-8B Textual)  

---

## Resumo

Este relatório apresenta a análise de compreensão teórica e o plano de ataque conceitual para a implantação e avaliação do **Pipeline 03**, um dos doze pipelines de inteligência artificial propostos para o diagnóstico e controle do escoamento bifásico (água-vapor) no Circuito de Circulação Natural (CCN) do IPEN. O Pipeline 03 investiga uma arquitetura modular desacoplada composta por três estágios: visão computacional clássica determinística via OpenCV para extração quantitativa de feições (Estágio 1); rede neural convolucional leve EfficientNet-B0 para classificação categórica de regimes e estados do fluido (Estágio 2); e um Modelo de Linguagem de Grande Porte puramente textual, o LLaMA-3-8B, atuando como motor de decisão e geração de pareceres operacionais (Estágio 3). O objetivo central deste pipeline é testar a hipótese de que a conversão de atributos visuais em dados numéricos e textuais estruturados via *prompting* permite que um LLM puramente textual desempenhe inferências operacionais corretas, dispensando o elevado custo computacional do processamento visual direto por modelos multimodais (VLMs). Descreve-se a fundamentação de cada módulo, o arranjo de quantização em 4-bit (NF4/AWQ) necessário para sustentação do sistema em hardware local restrito (NVIDIA RTX 4050 6GB VRAM), a engenharia de *prompt* estruturado e o protocolo de teste experimental.

---

## 1. Introdução e Contextualização do Doutorado

O estudo da termoidráulica de escoamentos bifásicos em regimes de circulação natural é fundamental para a segurança e otimização de reatores nucleares passivos, nos quais a remoção de calor residual ocorre sem a necessidade de bombeamento mecânico externo. A identificação precisa da topologia das fases (regimes borbulhante, pistonado/*slug*, anular e *churn*) e a estimativa contínua de parâmetros como a fração de vazio ($\alpha$) e a frequência de geração de bolhas são cruciais para detectar instabilidades termoidráulicas antes de transientes críticos.

No âmbito da proposta de doutorado desenvolvida junto ao IPEN/CNEN, busca-se avaliar a robustez, a latência e a capacidade de generalização de Modelos Fundamentais de Visão e Linguagem (*Vision-Language Models* - VLMs) em comparação com sistemas desacoplados e regras determinísticas. Dentro deste espectro, o **Pipeline 03** desempenha o papel de um ecossistema estatístico-linguístico híbrido. Em vez de passar os tensores de imagem diretamente a um modelo multimodal de alto peso, o Pipeline 03 transfere a responsabilidade do processamento geométrico para algoritmos leves de visão (OpenCV e EfficientNet-B0) e utiliza a capacidade de raciocínio lógico-linguístico do LLaMA-3-8B exclusivamente sobre abstrações textuais e métricas extraídas.

Além disso, a validação do Pipeline 03 subsidia metodologicamente o desenvolvimento do **benchmark multimodal** proposto no doutorado, fornecendo dados comparativos entre a inferência baseada exclusivamente em vetores numéricos/descritores e a inferência baseada em tensores visuais brutos.

---

## 2. Análise Técnica dos Estágios da Arquitetura

O Pipeline 03 é construído sob uma topologia em cascata sequencial. A figura conceitual a seguir ilustra o fluxo de processamento entre os três estágios:

```
+-----------------------------------------------------------------------------------+
|                                ESTÁGIO 1: OpenCV                                 |
| Frame RGB (Janela CCN) ----> Filtros / Canny / Contornos ----> [N_bolhas, Area_%] |
+---------------------------------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------+-------------------------------+
|                             ESTÁGIO 2: EfficientNet-B0                            |
| Frame RGB -------------> Deep ConvNet (Transfer Learning) ----> [Regime / Estado] |
+---------------------------------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------+-------------------------------+
|                           INJETOR DE PROMPT E ESTRUTURAÇÃO                         |
| Consolidação: Telemetria CCN + Dados OpenCV + Estado EfficientNet                 |
+---------------------------------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------+-------------------------------+
|                            ESTÁGIO 3: LLaMA-3-8B (INT4)                            |
| Text Prompt Quantizado -------> Inference Engine ------> Parecer / Ajuste de Vazão |
+-----------------------------------------------------------------------------------+
```

### 2.1. Estágio 1: Processamento Óptico Clássico com OpenCV

O Estágio 1 assume a função de pré-processar as imagens de alta velocidade capturadas na janela de visualização do tubo de circulação natural do CCN e calcular métricas determinísticas brutas em tempo real.

#### Algoritmos Utilizados
*   **Filtragem de Ruído e Pré-processamento:** Conversão para escala de cinza, equalização de histograma adaptativa limitada por contraste (CLAHE) para supressão de iluminação não uniforme no tubo de vidro, e suavização por filtro Gaussiano.
*   **Detecção de Bordas (Canny Edge Detector):** Aplicação dos gradientes de Sobel para localização de interfaces de fase líquido-gás, onde o salto de índice de refração produz altas variações de intensidade luminosa.
*   **Análise de Contornos e Binarização:** Limiarização de Otsu e algoritmos de busca de contornos para delimitação geométrica dos contornos fechados correspondentes às bolhas e bolsas de vapor.

#### Métricas Extraídas
1.  **Contagem de Bolhas ($N_{bolhas}$):** Quantidade total de contornos válidos identificados na região de interesse (ROI).
2.  **Área Total de Vapor e Fração de Vazio Óptica ($\alpha_{área}$):** Razão entre o número de pixels pertencentes a áreas delimitadas por bolhas ($A_{vapor}$) e a área total da seção transversal do tubo na ROI ($A_{tubo}$):

$$\alpha_{área} = \frac{\sum A_{vapor}}{A_{tubo}}$$

#### Vantagens e Limitações
*   *Vantagens:* Latência ultrabaixa ($< 3\text{ ms}$ por frame em CPU), baixo consumo de memória e caráter determinístico (não sujeito a alucinações).
*   *Limitações:* Alta sensibilidade a reflexos ópticos no tubo de vidro, sobreposição visual de bolhas em regimes de alta fração de vazio (*churn*) e falta de contexto semântico global do estado do escoamento.

### 2.2. Estágio 2: Classificação Estruturada de Estado com EfficientNet-B0

Enquanto o Estágio 1 quantifica feições geométricas locais, o Estágio 2 fornece a classificação semântica e categórica do estado do fluido utilizando a arquitetura convolucional **EfficientNet-B0** (Tan & Le, 2019).

#### Fundamentação Teórica
A EfficientNet-B0 utiliza o princípio de escalonamento composto (*compound scaling*), que equilibra uniformemente a profundidade ($d$), a largura de canais ($w$) e a resolução da imagem de entrada ($r$) por meio de um coeficiente composto $\phi$:

$$d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi \quad \text{sujeito a} \quad \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$$

Utilizando a rede base desenvolvida por busca de arquitetura neural (*Neural Architecture Search* - NAS), a EfficientNet-B0 utiliza blocos *Mobile Inverted Bottleneck Convolution* (MBConv) com módulos de *Squeeze-and-Excitation* (SE), alcançando alta acurácia na extração de padrões espaciais com apenas $\sim 5{,}3$ milhões de parâmetros e $\sim 0{,}39$ GFLOPs.

#### Papel no Pipeline
O modelo é submetido a *transfer learning* (ajuste fino nas camadas superiores) utilizando a base de imagens do CCN do IPEN, classificando o frame em 5 categorias fundamentais de regime de escoamento:
1.  *Single-Phase Liquid* (Monofásico)
2.  *Bubbly Flow* (Borbulhante)
3.  *Slug Flow* (Pistonado / Bolsas de Vapor)
4.  *Churn / Transicional* (Turbulento desordenado)
5.  *Annular Flow* (Anular com filme líquido de parede)

Além disso, a rede estima o vetor de probabilidade Softmax referente à fase transiente (ex.: estabilidade vs. iminência de expulsão de vapor).

### 2.3. Estágio 3: Raciocínio Operacional com LLaMA-3-8B (Puramente Textual)

O Estágio 3 é o motor analítico e executivo do pipeline, baseado no **LLaMA-3-8B** (Dubey et al., 2024 / AI@Meta).

#### Fundamentação do Modelo
O LLaMA-3-8B é uma arquitetura *Decoder-Only Transformer* de 8 bilhões de parâmetros treinada em mais de 15 trilhões de *tokens*. Ele apresenta um vocabulário otimizado de $128.000$ *tokens* (tiktoken) e atenção com consulta agrupada (*Grouped-Query Attention* - GQA), o que reduz drasticamente o footprint da memória de chave-valor (KV Cache) durante o raciocínio.

#### Estratégia de Quantização para Hardware de Borda (RTX 4050 6GB)
A execução de um modelo de 8B parâmetros em precisão FP16 exige aproximadamente $16\text{ GB}$ de VRAM apenas para alocação dos pesos, tornando-o inviável para a GPU dedicada de bancada (NVIDIA RTX 4050 Laptop com 6GB VRAM). Para contornar essa restrição, o Pipeline 03 adota estratégias de quantização pós-treinamento (*Post-Training Quantization* - PTQ):
*   **Formatos Avaliados:** INT4 (via GGUF / `llama.cpp` ou AWQ / Activation-aware Weight Quantization) e NF4 (NormalFloat 4-bit via `bitsandbytes` / QLoRA).
*   **Consumo de VRAM Estimado:**
    *   Pesos quantizados (4-bit): $\sim 4{,}2\text{ GB}$
    *   KV Cache e contexto ($2048$ tokens): $\sim 0{,}8\text{ GB}$
    *   Margem de segurança da GPU: $\sim 0{,}8\text{ GB}$
    *   *Total:* $\sim 5{,}8\text{ GB}$, enquadrando-se dentro do limite de 6GB da RTX 4050.

---

## 3. Conceito da Aplicação e Arquitetura do Sistema

### 3.1. Gerador Automatizado de Pareceres Operacionais e Ajuste de Vazão
O conceito central de aplicação do Pipeline 03 é a criação de um **Agente de Telemetria e Diagnóstico Físico-Operacional**. Em instalações nucleares e térmicas contendo circuitos de circulação natural, as decisões operacionais (como alteração da potência de aquecimento na seção de teste ou ajuste de válvulas de controle de vazão) dependem do correto diagnóstico da estabilidade do fluido.

O Pipeline 03 atua convertendo as variáveis físicas determinísticas extraídas do vídeo e a telemetria do circuito em um texto estruturado consumido pelo LLaMA-3-8B. O modelo não recebe imagens; ele recebe a "tradução matemática" da imagem.

### 3.2. Formatação do Prompt e Engenharia de Contexto
A integração entre os três estágios é realizada por meio de um *template* de *prompt* rígido formatado em Markdown/JSON:

```text
[SYSTEM PROMPT]
Você é um Engenheiro de Operação especialista na termoidráulica do Circuito de Circulação Natural (CCN) do IPEN.
Sua função é analisar as métricas quantitativas de visão computacional e telemetria e emitir um parecer técnico estrito, seguido de recomendação de controle de vazão.

[DADOS DE ENTRADA DO SISTEMA]
- Frequência de Amostragem: 60 FPS
- Contagem de Bolhas (OpenCV - Estágio 1): 47 bolhas/frame
- Fração de Vazio Estimada (OpenCV - Estágio 1): 28.4%
- Regime de Escoamento Identificado (EfficientNet-B0 - Estágio 2): Slug Flow (Confiança: 94.2%)
- Temperatura do Fluido (Sensor TC-01): 88.5 °C
- Pressão Diferencial (Sensor DP-02): 12.3 kPa
- Vazão Atual do Circuito: 0.35 L/min

[TAREFA]
1. Avalie se o sistema está em regime estável ou sob risco de instabilidade termoidráulica.
2. Forneça uma justificativa física baseada na correlação entre a fração de vazio e o regime Slug Flow.
3. Determine a ação operacional (Manter, Incrementar Vazão, Decrementar Potência ou Desligamento de Emergência).
4. Retorne a resposta estritamente no formato JSON estruturado.
```

---

## 4. Plano de Ataque Conceitual e Esquematização dos Testes

Para validar a hipótese do Pipeline 03 no contexto da tese de doutorado, é estabelecido um plano de ataque dividido em quatro etapas sequenciais:

```
+-----------------------------------------------------------------------------------+
| ETAPA 1: Construção do Pipeline de Extração e Quantização                         |
| - Integração do OpenCV e EfficientNet-B0 em PyTorch/C++                           |
| - Quantização do LLaMA-3-8B (AWQ / GGUF 4-bit) com vLLM ou llama.cpp              |
+---------------------------------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------+-------------------------------+
| ETAPA 2: Curadoria do Conjunto de Teste (Benchmark CCN)                           |
| - Seleção de 500 janelas temporais rotuladas (Casos Nominais + Casos OOD)         |
+---------------------------------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------+-------------------------------+
| ETAPA 3: Execução dos Protocolos de Inferência                                    |
| - Medição de Latência individual dos Estágios 1, 2 e 3                            |
| - Avaliação do Throughput (Tokens/segundo) e pico de VRAM                         |
+---------------------------------------------------+-------------------------------+
                                                    |
                                                    v
+---------------------------------------------------+-------------------------------+
| ETAPA 4: Validação Diagnóstica e Análise Qualitativa                              |
| - Avaliação de Acurácia Diagnóstica e verificação de alucinações no JSON         |
+-----------------------------------------------------------------------------------+
```

### 4.1. Protocolo de Avaliação de Desempenho

O teste do Pipeline 03 submeterá o sistema a 500 amostras do benchmark experimental do CCN, divididas em:
*   **Conjunto In-Distribution (ID):** Amostras com iluminação controlada e regimes bem definidos (350 amostras).
*   **Conjunto Out-of-Distribution (OOD):** Amostras com ruído óptico simulado (desfoque, variação severa de iluminação) e transitórios térmicos abruptos (150 amostras).

### 4.2. Métricas de Avaliação
1.  **Acurácia da Identificação de Regime (Estágio 2):** Matriz de confusão da EfficientNet-B0 contra os rótulos de especialistas humanos.
2.  **Precisão da Ação Operacional (Estágio 3):** Percentual de recomendações operacionais emitidas pelo LLaMA-3-8B idênticas à matriz de decisão de segurança recomendada para o CCN.
3.  **Latência Fim-a-Fim ($\Delta t_{total}$):**

$$\Delta t_{total} = \Delta t_{OpenCV} + \Delta t_{EfficientNet} + \Delta t_{Prompt} + \Delta t_{LLaMA\_TTFT} + \Delta t_{LLaMA\_Gen}$$

Onde $\Delta t_{LLaMA\_TTFT}$ representa o *Time-To-First-Token* e $\Delta t_{LLaMA\_Gen}$ o tempo de geração do relatório.

4.  **Taxa de Aderência JSON (Syntax Validity):** Medida da capacidade do LLaMA-3-8B quantizado em manter o formato estruturado solicitado sem violar esquemas.

---

## 5. Resultados Esperados e Análise Custo-Benefício

A partir das especificações do Pipeline 03, antecipam-se as seguintes conclusões experimentais e métricas de desempenho:

### 5.1. Perfil de Latência Esperado na RTX 4050
*   **Estágio 1 (OpenCV):** $\sim 2\text{ a }5\text{ ms}$ por frame (CPU/GPU).
*   **Estágio 2 (EfficientNet-B0):** $\sim 8\text{ a }12\text{ ms}$ por frame (Batch Size = 1 em GPU).
*   **Estágio 3 (LLaMA-3-8B 4-bit AWQ/GGUF):**
    *   *TTFT (Time-to-first-token):* $\sim 120\text{ a }250\text{ ms}$.
    *   *Geração de Texto (50 tokens de resposta JSON):* $\sim 1{,}5\text{ a }2{,}5\text{ s}$ ($\sim 25\text{ a }35\text{ tokens/s}$).
*   **Latência Total do Ciclo:** $\sim 1{,}7\text{ a }2{,}7\text{ segundos}$.

### 5.2. Análise Teórica e Desempenho Esperado

| Atributo | Estimativa do Pipeline 03 | Implicação para a Tese de Doutorado |
| :--- | :--- | :--- |
| **Uso de VRAM** | $\sim 5{,}2\text{ GB} - 5{,}8\text{ GB}$ | Viabiliza o uso em hardware de bancada (RTX 4050 6GB) sem requerer clusters de alto custo. |
| **Precisão Numérica** | Alta no Estágio 1 (Determinística) | Evita alucinações de contagem direta comuns em VLMs de visão profunda. |
| **Explicabilidade (XAI)** | Muito Alta no Estágio 3 | O LLaMA-3-8B gera pareceres textuais claros justificando a tomada de decisão. |
| **Robustez a Ruído Visual** | Moderada a Alta | Falhas de segmentação no Estágio 1 podem propagar erros para o Estágio 3 se não tratadas no prompt. |
| **Adequação ao Tempo Real** | Baixa a Média (Assíncrono) | O ciclo de resposta ($\sim 2\text{ s}$) é adequado para diagnósticos termoidráulicos operacionais, mas não para desligamento instantâneo de emergência ($< 10\text{ ms}$). |

### 5.3. Hipóteses para Comparação Multimodal
Espera-se que o Pipeline 03 apresente uma precisão quantitativa de área/contagem superior à de VLMs diretos e leves (como o Moondream2 do Pipeline 02), pois baseia seu raciocínio em números exatos calculados deterministicamente. Por outro lado, caso o Estágio 1 falhe severamente sob ruído (ex.: reflexos no tubo alterando a contagem de bolhas), o LLaMA-3-8B tenderá a tomar decisões operacionais incorretas embasadas nos dados numéricos corrompidos, evidenciando o fenômeno de *Propagação de Erro em Cascata* em sistemas desacoplados.

---

## 6. Considerações Finais

O **Pipeline 03** estabelece uma abordagem robusta, modular e explicável para a automação diagnóstica do Circuito de Circulação Natural do IPEN. A integração do processamento determinístico leve (OpenCV e EfficientNet-B0) com a capacidade lógicos-semântica de um LLM quantizado em 4-bit (LLaMA-3-8B) resolve o gargalo de alocação de memória mantendo o consumo de VRAM abaixo de 6GB.

O relatório de compreensão conclui que o Pipeline 03 é teoricamente e tecnicamente viável para implantação no hardware do projeto. O plano de ataque traçado subsidiará a execução da **Tarefa 02 (Desenvolvimento)** e da **Tarefa 03 (Conclusão e Testes)** previstas na agenda de trabalho.

---

## 7. Referências Bibliográficas

1.  **AI@Meta** (2024). *The Llama 3 Herd of Models*. arXiv preprint arXiv:2407.21783.
2.  **Bradski, G.** (2000). *The OpenCV Library*. Dr. Dobb's Journal of Software Tools, 25(3), 120-125.
3.  **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2024). *QLoRA: Efficient Finetuning of Quantized LLMs*. Advances in Neural Information Processing Systems (NeurIPS 2023), 36.
4.  **Lin, J., Tang, J., Yang, H., Zhang, Y., Xiao, S., Gan, C., & Han, S.** (2023). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*. arXiv preprint arXiv:2306.09782.
5.  **Tan, M., & Le, Q. V.** (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. Proceedings of the 36th International Conference on Machine Learning (ICML 2019), PMLR 97:6105-6114.
6.  **IPEN/CNEN** (2024). *Plano de Trabalho de Doutorado: Robustez e Generalização de Modelos Fundamentais de Visão e Linguagem em Escoamento Bifásico*. Instituto de Pesquisas Energéticas e Nucleares, São Paulo.