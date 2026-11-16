# Avaliação Experimental de Arquitetura Modular Desacoplada (OpenCV, EfficientNet-B0 e LLaMA-3-8B) para Diagnóstico Termo-Hidráulico de Escoamento Bifásico em Circuito de Circulação Natural

**Projeto de Doutorado:** Robustez e Generalização de Modelos Fundamentais de Visão e Linguagem em Escoamento Bifásico  
**Instituição:** Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN) — São Paulo, Brasil  
**Ambiente Experimental:** Circuito de Circulação Natural (CCN) — Centro de Engenharia Nuclear (CEN)  
**Documento:** Relatório Final de Desenvolvimento, Conclusão e Documentação — Pipeline 03  
**Responsável Técnico:** Heitor (Onda 1 — Cronograma Oficial de Pesquisa)  
**Data de Consolidação:** Novembro de 2026  

---

## Resumo

Este artigo apresenta os resultados experimentais, a modelagem matemática e a validação estatística do **Pipeline 03**, uma arquitetura híbrida e desacoplada proposta no escopo do projeto de doutorado do IPEN/CNEN para o monitoramento automatizado de escoamentos bifásicos (água-vapor) no Circuito de Circulação Natural (CCN). O sistema estrutura-se em três estágios sequenciais: (1) extração geométrica determinística via visão computacional clássica (OpenCV — algoritmo de Canny e integração topológica de contornos); (2) classificação profunda de estado termo-hidráulico (*Incubação* vs. *Expulsão*) mediante ajuste fino (*fine-tuning*) da rede convolucional EfficientNet-B0; e (3) geração de diretrizes operacionais de segurança por meio do Modelo de Linguagem de Grande Escala LLaMA-3-8B puramente textual, executado localmente sob quantização em 4 bits via servidor C++ Ollama. Confrontam-se as expectativas teóricas inicialmente projetadas para execução em GPU dedicada (NVIDIA RTX 4050) com a realidade operacional de implantação em hardware portátil restrito sem aceleração gráfica dedicada (Intel Core i5-1335U, gráficos integrados Intel Iris Xe e $16\text{ GB}$ de RAM compartilhada). Documenta-se o processo metodológico ao longo de três iterações experimentais, nas quais foi diagnosticado e corrigido o colapso de escala de tensores ($[0, 1] \to [0, 255]$) e o desvio de covariância interna das camadas *BatchNormalization* na EfficientNet-B0. No experimento consolidado (`UnifiedPipeline03`), aplicando-se um regime de escassez amostral de treinamento ($N_{\text{treino}} = 30$ imagens, $30\%$) contra validação massiva ($N_{\text{teste}} = 70$ imagens, $70\%$), o classificador atingiu acurácia global de $97{,}14\%$, precisão de $100{,}00\%$ para *Incubação*, revocação de $100{,}00\%$ para *Expulsão* (zero falsos negativos críticos) e Coeficiente de Correlação de Matthews $\text{MCC} = 0{,}9444$. A telemetria de sistema revelou latência média fim-a-fim de $9.106{,}19\text{ ms}$ ($\sigma = 1.219{,}57\text{ ms}$), sendo $0{,}03\%$ consumida pelo OpenCV ($2{,}98\text{ ms}$), $1{,}72\%$ pela EfficientNet-B0 ($156{,}63\text{ ms}$) e $98{,}25\%$ pela inferência autorregressiva do LLaMA-3-8B ($8.946{,}49\text{ ms}$). Por fim, comprova-se matematicamente o fenômeno de *Propagação de Erro em Cascata*: a incapacidade do detector de bordas de Canny em integrar o volume interno das fases gasosas limitou a fração de vazio óptica máxima a $12{,}07\%$, induzindo o LLaMA-3-8B a acionar protocolos de suspeita de falha de calibração de sensor em $100\%$ dos eventos de expulsão.

**Palavras-chave:** Escoamento Bifásico. Circulação Natural. Visão Computacional. EfficientNet-B0. LLaMA-3. Modelos Fundamentais. Engenharia Nuclear.

---

## 1. Introdução e Contextualização Científica

A segurança operacional de reatores nucleares avançados e sistemas térmicos passivos fundamenta-se na capacidade de manter a integridade da transferência de calor sob regimes transientes severos (ISHII; HIBIKI, 2011). No Circuito de Circulação Natural (CCN) instalado no Centro de Engenharia Nuclear do Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN), o escoamento ocorre sem o auxílio de bombas mecânicas ativas: a força motriz do fluido emerge exclusivamente do diferencial de densidade ($\Delta \rho$) induzido pelo gradiente térmico entre a seção de aquecimento (perna quente) e o trocador de calor superior (perna fria) (TORRES; TORRES, 2006; MASOTTI; DE MESQUITA, 2015).

Em condições de operação bifásica, a evolução da fase gasosa ao longo da janela óptica de visualização transita desde a nucleação incipiente e escoamento borbulhante disperso (*Incubação*) até a formação de grandes bolsas de gás (*slugs*), escoamento caótico (*churn*) e rápida descarga bifásica (*Expulsão*) (ISHII; HIBIKI, 2011; LUO et al., 2023). A detecção tardia da transição entre incubação e expulsão pode acarretar instabilidades termo-hidráulicas do tipo gêiser (*geysering*), oscilações de ondas de densidade e crise de ebulição com secagem da parede aquecida (*dryout*).

Recentemente, a literatura especializada tem migrado de métodos puramente empíricos para arquiteturas de aprendizado profundo (*Deep Learning*), empregando Redes Neurais Convolucionais (CNNs) para classificação espacial de padrões de escoamento (DIAS et al., 2025; JIANG et al., 2022) e Redes Neurais Recorrentes (LSTMs) para séries temporais (ZHANG; WANG; LI, 2023). Paralelamente, a consolidação dos Modelos Fundamentais (*Foundation Models* — FMs) (BOMMASANI et al., 2021; YANG et al., 2022), abrangendo Modelos de Visão e Linguagem (*Vision-Language Models* — VLMs) como CLIP (RADFORD et al., 2021) e Flamingo (ALAYRAC et al., 2022), inaugurou a possibilidade de unir percepção visual à interpretação semântica em linguagem natural.

No entanto, a aplicação direta de modelos multimodais densos enfrenta três barreiras críticas na engenharia nuclear experimental:
1. **Elevado custo computacional e de memória VRAM**, inviabilizando a inferência local em dispositivos de borda ou computadores de bancada (QIAO et al., 2023);
2. **Alucinações quantitativas**, nas quais VLMs generalistas falham ao estimar grandezas numéricas exatas (como contagem de bolhas e fração de vazio planar);
3. **Vulnerabilidade a perturbações e dados fora da distribuição (*Out-of-Distribution* — OOD)** (MADRY et al., 2018; HUA et al., 2024; QUINTINO; STANKO, 2025).

Para investigar sistematicamente esses limites, o plano de doutorado do IPEN/CNEN estruturou uma matriz experimental de **12 Pipelines Híbridos**. Dentro desta matriz, o **Pipeline 03** foi concebido para testar uma hipótese científica específica: **avaliar se um Modelo de Linguagem de Grande Escala puramente textual (LLaMA-3-8B), privado de visão direta da imagem, é capaz de inferir diagnósticos e diretrizes operacionais precisas quando alimentado exclusivamente pela tradução matemática do escoamento extraída via visão computacional clássica (OpenCV) e classificação convolucional eficiente (EfficientNet-B0)**.

---

## 2. Expectativa Teórica vs. Realidade de Implantação Operacional

Na fase preliminar do estudo (*Tarefa 01 — Relatório de Compreensão*), o dimensionamento do Pipeline 03 foi projetado assumindo a disponibilidade imediata de uma estação de trabalho equipada com placa aceleradora dedicada **NVIDIA GeForce RTX 4050 ($6\text{ GB}$ GDDR6 VRAM)**. Contudo, a execução prática durante a sprint de desenvolvimento foi conduzida na estação móvel real do pesquisador (**HEITOR-GLXBOOK4**), cujas especificações impuseram restrições severas de arquitetura de computador:

* **Unidade Central de Processamento (CPU):** $13^{\text{th}}\text{ Gen}$ Intel Core i5-1335U ($1{,}30\text{ GHz}$ base, arquitetura híbrida de $10\text{ núcleos}$ físicos [$2\text{ P-Cores} + 8\text{ E-Cores}$] e $12\text{ threads}$ lógicas, instruções AVX2 e AVX-VNNI ativas);
* **Memória Principal (RAM):** $16{,}0\text{ GB}$ LPDDR4x ($15{,}7\text{ GB}$ utilizáveis em arquitetura de memória unificada/compartilhada);
* **Unidade de Processamento Gráfico (GPU):** Integrada Intel Iris Xe Graphics ($128\text{ MB}$ de memória dedicada reservada, sem suporte nativo à pilha NVIDIA CUDA/cuDNN);
* **Ambiente de Software:** Windows 64 bits nativo, Python $3.13.12$, TensorFlow $2.18+$ com otimizações `oneDNN` em CPU e servidor de inferência local `Ollama` ($0.5+$) baseado em `llama.cpp`.

A Tabela 1 sintetiza o confronto direto entre os parâmetros projetados no estudo de viabilidade teórica e as métricas reais observadas após as adaptações de engenharia de software.

**Tabela 1 — Comparativo entre a Expectativa Teórica (Tarefa 01) e a Realidade Experimental (Tarefas 02 e 03)**

| Parâmetro de Projeto | Expectativa Teórica (Relatório de Compreensão) | Realidade Experimental Consolidada (`UnifiedPipeline03`) | Desvio / Adaptação de Engenharia Realizada |
| :--- | :--- | :--- | :--- |
| **Acelerador de Hardware** | GPU Dedicada NVIDIA RTX 4050 ($6\text{ GB}$ VRAM) | CPU Intel Core i5-1335U ($10\text{C}/12\text{T}$) + $16\text{ GB}$ RAM | Migração total para execução em CPU com instruções vetoriais AVX-VNNI e `oneDNN`. |
| **Motor de Inferência do LLM** | PyTorch / HuggingFace `transformers` (AWQ / NF4) | Servidor C++ `Ollama` (`llama.cpp` GGUF INT4) via HTTP | Isolamento de processo via API REST local para contornar incompatibilidades do `torch` no Python $3.13$. |
| **Latência Estágio 1 (OpenCV)** | $2{,}00\text{ ms a }5{,}00\text{ ms}$ | Média: $2{,}98\text{ ms}$ (Mediana: $2{,}41\text{ ms}$) | **Dentro do esperado.** Custo computacional marginal ($0{,}03\%$ do ciclo). |
| **Latência Estágio 2 (EfficientNet)** | $8{,}00\text{ ms a }12{,}00\text{ ms}$ (em GPU CUDA) | Média: $156{,}63\text{ ms}$ (Mediana: $149{,}20\text{ ms}$ em CPU) | Aumento de $\sim 13\times$ devido à ausência de núcleos tensores CUDA, mas ainda ágil ($\sim 6{,}4\text{ FPS}$). |
| **Latência Estágio 3 (LLaMA-3-8B)** | $1.500\text{ ms a }2.500\text{ ms}$ ($25\text{ a }35\text{ tokens/s}$) | Média: $8.946{,}49\text{ ms}$ ($\sim 4{,}5\text{ a }5{,}5\text{ tokens/s}$) | Gargalo de banda de memória RAM DDR4 ($\sim 50\text{ GB/s}$) frente à VRAM GDDR6 ($\sim 192\text{ GB/s}$). |
| **Latência Fim-a-Fim ($\Delta t_{\text{total}}$)** | $1.700\text{ ms a }2.700\text{ ms}$ | Média: $9.106{,}19\text{ ms}$ (Mediana: $9.210{,}86\text{ ms}$) | Aumento de $\sim 3{,}8\times$, mantendo viabilidade para supervisão assíncrona de bancada. |
| **Consumo de Memória** | $5{,}2\text{ GB a }5{,}8\text{ GB}$ VRAM | $87{,}13\%$ da RAM do Sistema ($\sim 13{,}7\text{ GB}$ totais) | Coexistência do S.O. Windows, tensores Keras/TF e pesos GGUF ($4{,}7\text{ GB}$) na mesma RAM física. |
| **Qualidade da Área do OpenCV** | Esperada alta precisão determinística | Subestimação severa: média de $4{,}39\%$ (máx. $12{,}07\%$) | Falha física do filtro de Canny em contabilizar o interior das bolsas de vapor. |

---

## 3. Metodologia Experimental e Evolução Iterativa

A construção do Pipeline 03 não ocorreu de maneira monolítica, mas seguiu um método científico empírico-indutivo estruturado em **três iterações sucessivas (Gerações 1, 2 e 3)**. Cada ciclo experimental expôs falhas numéricas, gargalos de sistema operacional ou anomalias de aprendizado de máquina que foram diagnosticados e solucionados na iteração subsequente.

```
+=========================================================================================+
|                  LINHA DO TEMPO METODOLÓGICA DE DESENVOLVIMENTO (P03)                   |
+=========================================================================================+
|                                                                                         |
|  [ITERAÇÃO 1: Protótipo Base com Simulador]                                             |
|   * Treino: 80 imgs (5 épocas, base EfficientNet 100% congelada, sem camada Rescaling)  |
|   * Teste: 20 imgs | LLM: Simulador lógico estático                                     |
|   * Diagnóstico: Cold Start de 2.598,62 ms no Teste 0; Acurácia colapsada em 40,00%.    |
|                                           |                                             |
|                                           v                                             |
|  [ITERAÇÃO 2: Integração Real Ollama + Descongelamento Parcial Bruto]                   |
|   * Treino: 80 imgs (10 épocas, 20 camadas descongeladas incluindo BatchNormalization)  |
|   * Teste: 20 imgs | LLM: LLaMA-3-8B real via HTTP POST (sem sessão persistente)        |
|   * Diagnóstico: Cold Start do Ollama de 23.835,54 ms; Acurácia aparente de 60,00%,     |
|     porém com 100% de viés na classe Expulsão devido a tensores em [0, 1] sem Rescaling.|
|                                           |                                             |
|                                           v                                             |
|  [ITERAÇÃO 3: Arquitetura Consolidada UnifiedPipeline03 + Telemetria]                   |
|   * Partição Invertida (Low-Data Regime): 30% Treino (30 imgs) / 70% Teste (70 imgs)    |
|   * Correções: Camada Rescaling(255.0) embutida + BatchNormalization congelada +        |
|     Cache binário local (.npz) + Rotina de Warm-up + Sessão HTTP Keep-Alive             |
|   * Resultado Final: Acurácia de 97,14% (68/70), zero Cold Start, telemetria completa.  |
|                                                                                         |
+=========================================================================================+
```

### 3.1. Iteração 1: Prototipagem Inicial e Falha de Transfer Learning Raso
Na primeira execução, o conjunto de $100\text{ imagens}$ ($50$ de *Expulsão* e $50$ de *Incubação*) foi dividido na proporção clássica de $80\%$ para treino ($80\text{ imagens}$) e $20\%$ para teste ($20\text{ imagens}$). Devido a conflitos de importação circular da biblioteca `torch` no Python $3.13$ nativo em Windows, o Estágio 3 foi temporariamente emulado por uma árvore de decisão sintética, enquanto a `EfficientNetB0` foi treinada por $5\text{ épocas}$ com todo o seu *backbone* convolucional congelado (`efficient_base.trainable = False`).

Dois problemas fundamentais emergiram:
1. **Efeito *Cold Start* do Grafo TensorFlow:** A primeira amostra (`ID_Teste 0`) exigiu $2.598{,}62\text{ ms}$ para compilação inicial dos operadores `oneDNN` em memória, distorcendo a latência média para $269{,}99\text{ ms}$, embora a mediana operacional real fosse de apenas $146{,}56\text{ ms}$.
2. **Incompatibilidade de Domínio Visual ($40{,}00\%$ de Acurácia):** Os filtros convolucionais congelados do ImageNet falharam em extrair características discriminativas das interfaces gás-líquido.

### 3.2. Iteração 2: Acoplamento do Servidor Ollama e Diagnóstico do "Bug Silencioso" de Escala
Na segunda geração, solucionou-se o impasse do Estágio 3 implantando o motor **Ollama** em segundo plano (porta TCP `11434`) com o modelo `llama3` ($8\text{B}$ parâmetros quantizado em $4\text{ bits}$). No Estágio 2, descongelaram-se as últimas $20\text{ camadas}$ da `EfficientNetB0` e treinou-se a rede por $10\text{ épocas}$ com taxa de aprendizado $\eta = 1 \times 10^{-4}$.

Os logs de treinamento revelaram estagnação patológica: a acurácia de treino oscilou em torno de $0{,}5250$ com perda (*loss*) travada em $0{,}6868$ (próxima a $\ln(2) \approx 0{,}6931$, valor teórico de um classificador binário puramente aleatório). Na execução do benchmark sobre $20\text{ imagens}$, a acurácia reportada foi de $60{,}00\%$, mas a inspeção da matriz de predições revelou que a rede classificou **todas as $20\text{ amostras}$ como *Expulsão***.

A auditoria matemática do código identificou a dupla causa raiz desse colapso:
* **Sub-escalonamento de Intensidade Luminosa:** O módulo utilitário do projeto (`utls_work_images.py`) normaliza os pixels para o intervalo contínuo $[0{,}0;\, 1{,}0]$ dividindo a matriz por $255{,}0$. Todavia, a implementação da família `EfficientNet` no Keras (`tensorflow.keras.applications.EfficientNetB0`) incorpora internamente uma camada inicial `Rescaling(1./255.)`, esperando tensores brutos no domínio $[0{,}0;\, 255{,}0]$. Ao receber matrizes já divididas por $255{,}0$, a rede operou sobre valores comprimidos no intervalo $[0{,}0;\, 0{,}00392]$, aniquilando a relação sinal-ruído dos gradientes.
* **Desvio de Covariância nas Camadas `BatchNormalization`:** Ao definir `layer.trainable = True` nas últimas $20\text{ camadas}$ sem filtrar as camadas de normalização em lote (IOFFE; SZEGEDY, 2015), o modelo atualizou as médias e variâncias móveis com lotes pequenos, desestabilizando os pesos pré-treinados.

Adicionalmente, no Estágio 3, a ausência de pré-aquecimento fez o Teste 1 sofrer um *Cold Start* de $23.835{,}54\text{ ms}$ enquanto o Ollama carregava os $4{,}7\text{ GB}$ de pesos do SSD para a RAM, estabilizando posteriormente em uma média de $9.360{,}66\text{ ms}$ nos testes 2 a 20.

### 3.3. Iteração 3: Consolidação na Arquitetura `UnifiedPipeline03`
Com base nos diagnósticos anteriores, projetou-se a arquitetura definitiva **`UnifiedPipeline03`**, introduzindo cinco inovações metodológicas alinhadas aos objetivos de doutorado:
1. **Inversão da Partição para Regime de Escassez de Dados (*Low-Data Regime*):** Reduziu-se o conjunto de treinamento para apenas **$30\%$ do acervo ($N_{\text{treino}} = 30$, sendo $15$ de *Expulsão* e $15$ de *Incubação*)**, reservando **$70\%$ ($N_{\text{teste}} = 70$, sendo $35$ de cada classe)** para validação rigorosa de generalização, com persistência em cache binário comprimido (`dados_teste_p3.npz`) para eliminar latência de rede e vazamento amostral (*data leakage*).
2. **Camada de Reescalonamento Inverso Embutida:** Inserção de uma camada `Rescaling(255.0)` no topo do grafo `Sequential`, restaurando a distribuição original $[0{,}0;\, 255{,}0]$ exigida pelo *backbone* da EfficientNet-B0.
3. **Congelamento Seletivo de `BatchNormalization`:** Descongelamento das últimas $20\text{ camadas}$ convolucionais mantendo estritamente `layer.trainable = False` para todas as instâncias de `BatchNormalization`.
4. **Protocolo de *Warm-up* Duplo:** Execução de uma inferência fantasma com tensor nulo na EfficientNet-B0 e um *ping* de geração curta (`num_predict: 5`) no Ollama antes do início da cronometragem oficial.
5. **Instrumentação de Telemetria de Hardware por Estágio:** Acoplamento da biblioteca `psutil` e cronômetros de alta resolução (`time.perf_counter()`) para registrar individualmente as latências dos Estágios 1, 2 e 3, bem como a carga percentual de CPU e ocupação de memória RAM a cada imagem processada.

---

## 4. Formulação Matemática dos Estágios do Pipeline 03

### 4.1. Estágio 1: Extração Geométrica Determinística via OpenCV
Seja $\mathbf{I}_{\text{norm}} \in [0, 1]^{H \times W \times 3}$ a matriz tridimensional normalizada proveniente do banco de dados, com altura $H = 224$ e largura $W = 224$. Primeiramente, reconstrói-se a matriz inteira de $8\text{ bits}$ $\mathbf{I} = \lfloor 255 \cdot \mathbf{I}_{\text{norm}} \rfloor \in \{0, \dots, 255\}^{224 \times 224 \times 3}$ e aplica-se a transformação colorimétrica ITU-R BT.601 para obtenção da luminância em escala de cinza $\mathbf{G}$:

$$\mathbf{G}(x, y) = 0{,}299 \cdot \mathbf{I}_R(x, y) + 0{,}587 \cdot \mathbf{I}_G(x, y) + 0{,}114 \cdot \mathbf{I}_B(x, y)$$

O detector de bordas de Canny (CANNY, 1986; BRADSKI, 2000) suaviza a imagem com um filtro Gaussiano bidimensional e calcula as derivadas direcionais por meio dos operadores de convolução de Sobel $\mathbf{K}_x$ e $\mathbf{K}_y$:

$$\mathbf{G}_x = \mathbf{G} * \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad \mathbf{G}_y = \mathbf{G} * \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

A magnitude do gradiente de borda $\mathbf{M}(x, y)$ e a direção normal à interface gás-líquido $\mathbf{\Theta}(x, y)$ são dadas por:

$$\mathbf{M}(x, y) = \sqrt{\mathbf{G}_x(x, y)^2 + \mathbf{G}_y(x, y)^2}, \quad \mathbf{\Theta}(x, y) = \arctan2\left(\mathbf{G}_y(x, y),\, \mathbf{G}_x(x, y)\right)$$

Após a supressão não-máxima na direção de $\mathbf{\Theta}(x, y)$, aplica-se a limiarização por histerese dupla com $T_{\text{min}} = 50$ e $T_{\text{max}} = 150$. Do mapa binário resultante, o algoritmo topológico de Suzuki e Abe (`cv2.findContours` com modo `RETR_EXTERNAL`) extrai o conjunto de contornos externos $\mathcal{C} = \{C_1, C_2, \dots, C_K\}$. A contagem discreta de bolhas é definida pela cardinalidade $N_{\text{bolhas}} = |\mathcal{C}| = K$.

A área planar de cada contorno $C_k$ (composto por $n_k$ vértices poligonais $(x_i, y_i)$) é calculada pela fórmula de Gauss (*Shoelace Formula*), equivalente discreto do Teorema de Green:

$$A(C_k) = \frac{1}{2} \left| \sum_{i=0}^{n_k - 1} \left( x_i y_{i+1} - x_{i+1} y_i \right) \right|$$

Por fim, o percentual de área de vapor ($\alpha_{\text{área}}$) na janela de $224 \times 224\text{ pixels}$ ($A_{\text{total}} = 50.176\text{ px}^2$) é expresso por:

$$\alpha_{\text{área}} = \left( \frac{\sum_{k=1}^{K} A(C_k)}{H \times W} \right) \times 100\%$$

### 4.2. Estágio 2: Classificação Profunda via EfficientNet-B0 Corrigida
A arquitetura EfficientNet-B0 (TAN; LE, 2019) dimensiona conjuntamente profundidade ($d = \alpha^\phi$), largura de canais ($w = \beta^\phi$) e resolução ($r = \gamma^\phi$) a partir do coeficiente base $\phi = 0$, utilizando blocos convolucionais invertidos (*MBConv*) com ativação *Swish* ($\text{Swish}(z) = z \cdot \sigma(z)$) e atenção de canais *Squeeze-and-Excitation*.

Para corrigir o colapso de escala na implantação com TensorFlow/Keras, o tensor de entrada $\mathbf{X} \in [0, 1]^{224 \times 224 \times 3}$ passa pela transformação linear afim de reescalonamento antes de alimentar o extrator de características $\mathcal{F}_{\text{EffNet}}$:

$$\tilde{\mathbf{X}} = 255{,}0 \cdot \mathbf{X}, \quad \mathbf{h} = \text{GAP}\left(\mathcal{F}_{\text{EffNet}}(\tilde{\mathbf{X}};\, \mathbf{\Theta}_{\text{frozen}},\, \mathbf{\Theta}_{\text{top20}})\right) \in \mathbb{R}^{1280}$$

onde $\text{GAP}$ denota o *Global Average Pooling* espacial bidimensional:

$$h_c = \frac{1}{7 \times 7} \sum_{u=1}^{7} \sum_{v=1}^{7} F_{u, v, c}, \quad \forall c \in \{1, \dots, 1280\}$$

Durante o ajuste fino nas $20\text{ camadas}$ superiores ($\mathbf{\Theta}_{\text{top20}}$), todas as camadas de *Batch Normalization* (IOFFE; SZEGEDY, 2015) permanecem em modo de inferência estrito, utilizando as médias $\boldsymbol{\mu}_{\text{pop}}$ e variâncias populacionais $\boldsymbol{\sigma}_{\text{pop}}^2$ pré-calculadas no ImageNet:

$$\hat{\mathbf{z}} = \boldsymbol{\gamma} \odot \frac{\mathbf{z} - \boldsymbol{\mu}_{\text{pop}}}{\sqrt{\boldsymbol{\sigma}_{\text{pop}}^2 + \epsilon}} + \boldsymbol{\beta}$$

A cabeça classificadora aplica regularização por *Dropout* ($p_{\text{drop}} = 0{,}3$ no treinamento), seguida por uma camada densa intermediária com ativação $\text{ReLU}$ ($\mathbf{W}_1 \in \mathbb{R}^{64 \times 1280}$) e neurônio de saída sigmoidal ($\mathbf{w}_2 \in \mathbb{R}^{1 \times 64}$):

$$\hat{p} = P(y = 1 \mid \mathbf{X}) = \sigma\left( \mathbf{w}_2 \, \max\left(\mathbf{0},\, \mathbf{W}_1 \mathbf{h} + \mathbf{b}_1\right) + b_2 \right) = \frac{1}{1 + e^{-z_{\text{out}}}}$$

A regra de decisão atribui o estado $\hat{y} = 1$ (*Incubação*) se $\hat{p} > 0{,}5$, e $\hat{y} = 0$ (*Expulsão*) se $\hat{p} \le 0{,}5$. O treinamento minimiza a função de perda de Entropia Cruzada Binária (*Binary Cross-Entropy* — BCE) sobre os $N_{\text{treino}} = 30$ exemplos via otimizador Adam (KINGMA; BA, 2015) com $\eta = 3 \times 10^{-4}$:

$$\mathcal{L}_{\text{BCE}}(\mathbf{\Theta}) = -\frac{1}{N_{\text{treino}}} \sum_{i=1}^{N_{\text{treino}}} \left[ y_i \ln(\hat{p}_i) + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$

### 4.3. Estágio 3: Inferência Textual Quantizada com LLaMA-3-8B
No Estágio 3, as variáveis extraídas ($\hat{y}$, $N_{\text{bolhas}}$, $\alpha_{\text{área}}$) são compiladas em uma sequência de tokens de entrada $\mathbf{x} = (x_1, x_2, \dots, x_M)$ através de uma função de formatação de *prompt* $\Phi(\hat{y}, N_{\text{bolhas}}, \alpha_{\text{área}})$. O modelo LLaMA-3-8B (DUBEY et al., 2024) estima a distribuição de probabilidade condicional da sequência de decisão operacional $\mathbf{w} = (w_1, \dots, w_T)$ de forma autorregressiva:

$$P(\mathbf{w} \mid \mathbf{x}) = \prod_{t=1}^{T} P\left(w_t \mid w_{<t},\, \mathbf{x};\, \mathbf{\Theta}_{\text{INT4}}\right) = \prod_{t=1}^{T} \frac{\exp\left(z_{t, w_t} / \tau\right)}{\sum_{v \in \mathcal{V}} \exp\left(z_{t, v} / \tau\right)}$$

onde $|\mathcal{V}| = 128.256$ é o tamanho do vocabulário, $\tau = 0{,}1$ é o hiperparâmetro de temperatura (forçando comportamento quase determinístico *greedy*) e $T \le 45$ é o teto de geração (`num_predict`).

Para viabilizar a execução nos $16\text{ GB}$ de RAM do sistema, os pesos originais em ponto flutuante de $16\text{ bits}$ ($\mathbf{W}_{\text{FP16}}$) são quantizados por blocos em inteiros de $4\text{ bits}$ ($\mathbf{W}_{\text{INT4}} \in \{-8, \dots, 7\}$, formato `Q4_0`/`Q4_K_M` do `llama.cpp`) (DETTMERS et al., 2023; LIN et al., 2024):

$$\mathbf{W}_{\text{INT4}} = \text{round}\left( \frac{\mathbf{W}_{\text{FP16}}}{\Delta_B} \right), \quad \Delta_B = \frac{\max(|\mathbf{W}_B|)}{7}$$

Como a geração autorregressiva de tokens na CPU é limitada pela largura de banda da memória RAM (*Memory-Bandwidth Bound*), o tempo de geração de cada novo token ($\Delta t_{\text{token}}$) é governado aproximadamente pela razão entre o volume de pesos lidos da RAM ($M_{\text{pesos}} \approx 4{,}7\text{ GB}$) e a banda efetiva do controlador de memória LPDDR4x ($BW_{\text{RAM}} \approx 30\text{ a }40\text{ GB/s}$ reais em multiprocessamento):

$$\Delta t_{\text{Estágio 3}} = \Delta t_{\text{HTTP}} + \Delta t_{\text{Prefill}}(M) + \sum_{t=1}^{T_{\text{gerados}}} \left( \frac{M_{\text{pesos}} + M_{\text{KV}}(M+t)}{BW_{\text{RAM}}} + \Delta t_{\text{compute}} \right)$$

---

## 5. Resultados Experimentais e Análise Estatística Profunda

### 5.1. Convergência do Treinamento Supervisionado em Regime de Escassez ($N_{\text{treino}} = 30$)
O treinamento da `EfficientNet-B0` corrigida foi executado ao longo de $12\text{ épocas}$ com tamanho de lote (*batch size*) igual a $8$ ($4\text{ passos por época}$). A Tabela 2 e a Figura 1 detalham a rápida convergência da função de perda $\mathcal{L}_{\text{BCE}}$ e a estabilização da acurácia.

**Tabela 2 — Evolução Época a Época do Treinamento da EfficientNet-B0 Corrigida (CPU i5-1335U)**

| Época | Tempo Total (s) | Tempo por Passo (ms) | Loss de Treino ($\mathcal{L}_{\text{train}}$) | Acurácia de Treino ($\%$) | Loss de Validação ($\mathcal{L}_{\text{val}}$) | Acurácia de Validação ($\%$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | $12\text{ s}$ | $1.000\text{ ms}$ | $0{,}4631$ | $90{,}00\%$ | $0{,}20070$ | $100{,}00\%$ |
| **02** | $1\text{ s}$ | $381\text{ ms}$ | $0{,}1378$ | $100{,}00\%$ | $0{,}05120$ | $100{,}00\%$ |
| **03** | $1\text{ s}$ | $325\text{ ms}$ | $0{,}0946$ | $96{,}67\%$ | $0{,}01490$ | $100{,}00\%$ |
| **04** | $1\text{ s}$ | $301\text{ ms}$ | $0{,}0549$ | $96{,}67\%$ | $0{,}00700$ | $100{,}00\%$ |
| **05** | $1\text{ s}$ | $338\text{ ms}$ | $0{,}1024$ | $96{,}67\%$ | $0{,}01310$ | $100{,}00\%$ |
| **06** | $1\text{ s}$ | $384\text{ ms}$ | $0{,}0318$ | $96{,}67\%$ | $0{,}00140$ | $100{,}00\%$ |
| **07** | $1\text{ s}$ | $334\text{ ms}$ | $0{,}0037$ | $100{,}00\%$ | $0{,}00370$ | $100{,}00\%$ |
| **08** | $1\text{ s}$ | $366\text{ ms}$ | $0{,}0529$ | $96{,}67\%$ | $0{,}00180$ | $100{,}00\%$ |
| **09** | $1\text{ s}$ | $359\text{ ms}$ | $0{,}0024$ | $100{,}00\%$ | $0{,}00079$ | $100{,}00\%$ |
| **10** | $1\text{ s}$ | $311\text{ ms}$ | $0{,}0024$ | $100{,}00\%$ | $0{,}00061$ | $100{,}00\%$ |
| **11** | $1\text{ s}$ | $263\text{ ms}$ | $0{,}0050$ | $100{,}00\%$ | $0{,}00054$ | $100{,}00\%$ |
| **12** | $1\text{ s}$ | $241\text{ ms}$ | $0{,}0018$ | $100{,}00\%$ | $0{,}00041$ | $100{,}00\%$ |

![alt text](/docs/pt_br/pipeline_03/images/figura01_pipeline03.jpg)

Observa-se que a inserção da camada `Rescaling(255.0)` e o travamento das estatísticas de `BatchNormalization` reduziram a perda de validação em **três ordens de grandeza** (de $0{,}20070$ na Época 1 para $4{,}1129 \times 10^{-4}$ na Época 12), consumindo apenas $23\text{ segundos}$ totais de processamento em CPU.

---

### 5.2. Desempenho Preditivo e Generalização no Lote Massivo de Teste ($N_{\text{teste}} = 70$)
Submetendo o modelo treinado às $70\text{ imagens}$ inéditas do conjunto de teste ($35$ de *Expulsão* e $35$ de *Incubação*), obteve-se a Matriz de Confusão ilustrada na Figura 2.

![alt text](/docs/pt_br/pipeline_03/images/figura02_pipeline03.jpg)

*(Nota de convenção: Adotando a codificação nativa do dataset em que $y=0$ representa Expulsão e $y=1$ representa Incubação, calculam-se abaixo os indicadores analíticos para ambas as classes e os índices globais).*

#### Cálculos Analíticos das Métricas de Classificação:
1. **Acurácia Global ($\text{Acc}$):**
   $$\text{Acc} = \frac{TP_0 + TP_1}{N_{\text{teste}}} = \frac{35 + 33}{70} = \frac{68}{70} = 0{,}971428 \implies \mathbf{97{,}14\%}$$

2. **Métricas para a Classe Crítica de Segurança — *Expulsão* ($y = 0$):**
   * **Precisão ($\text{Prec}_0$):**
     $$\text{Prec}_0 = \frac{35}{35 + 2} = \frac{35}{37} = 0{,}945946 \implies \mathbf{94{,}59\%}$$
   * **Revocação / Sensibilidade ($\text{Rec}_0$):**
     $$\text{Rec}_0 = \frac{35}{35 + 0} = \frac{35}{35} = 1{,}000000 \implies \mathbf{100{,}00\%}$$
   * **Pontuação $F_1$ ($F_{1,\text{Expulsão}}$):**
     $$F_{1,\text{Expulsão}} = 2 \cdot \frac{\text{Prec}_0 \cdot \text{Rec}_0}{\text{Prec}_0 + \text{Rec}_0} = \frac{2 \times 35}{2 \times 35 + 2 + 0} = \frac{70}{72} = 0{,}972222 \implies \mathbf{97{,}22\%}$$

3. **Métricas para a Classe de Operação Nominal — *Incubação* ($y = 1$):**
   * **Precisão ($\text{Prec}_1$):**
     $$\text{Prec}_1 = \frac{33}{33 + 0} = \frac{33}{33} = 1{,}000000 \implies \mathbf{100{,}00\%}$$
   * **Revocação / Sensibilidade ($\text{Rec}_1$):**
     $$\text{Rec}_1 = \frac{33}{33 + 2} = \frac{33}{35} = 0{,}942857 \implies \mathbf{94{,}29\%}$$
   * **Pontuação $F_1$ ($F_{1,\text{Incubação}}$):**
     $$F_{1,\text{Incubação}} = 2 \cdot \frac{\text{Prec}_1 \cdot \text{Rec}_1}{\text{Prec}_1 + \text{Rec}_1} = \frac{2 \times 33}{2 \times 33 + 0 + 2} = \frac{66}{68} = 0{,}970588 \implies \mathbf{97{,}06\%}$$

4. **Coeficiente de Correlação de Matthews ($\text{MCC}$):**
   $$\text{MCC} = \frac{(35 \times 33) - (0 \times 2)}{\sqrt{(35 + 2)(35 + 0)(33 + 0)(33 + 2)}} = \frac{1.155}{\sqrt{37 \times 35 \times 33 \times 35}} = \frac{1.155}{35\sqrt{1.221}} = \frac{33}{34{,}94281} = \mathbf{0{,}9444}$$

#### Análise Física dos Outliers de Classificação (`ID_Teste 40` e `ID_Teste 60`):
Das $70\text{ imagens}$ avaliadas, o modelo cometeu apenas $2\text{ erros}$:
* **`ID_Teste 40` (41/70):** Rótulo Real = *Incubação*; Predição = *Expulsão* ($\hat{p} = 0{,}0488$). Nesta imagem, o OpenCV contabilizou **$486\text{ contornos de bolhas}$** e **$7{,}85\%$ de área** (muito acima da média de incubação de $159{,}8\text{ bolhas}$).
* **`ID_Teste 60` (61/70):** Rótulo Real = *Incubação*; Predição = *Expulsão* ($\hat{p} = 0{,}0042$). Nesta amostra, o OpenCV detectou **$369\text{ bolhas}$** e **$6{,}49\%$ de área**.

Do ponto de vista da engenharia de segurança nuclear, o fato de a rede apresentar **$100{,}00\%$ de revocação em *Expulsão*** (nenhum evento crítico real passou despercebido) e errar apenas em duas amostras de *Incubação* altamente turbulentas (pré-transição para expulsão) demonstra um comportamento **conservador pró-segurança (*fail-safe*)**. Ademais, nos $68\text{ acertos}$, a separação de confiança foi extrema: apenas $3\text{ amostras}$ corretas apresentaram incerteza marginal (`ID_Teste 33` com $\hat{p} = 0{,}0208$, `ID_Teste 56` com $\hat{p} = 0{,}0301$ e `ID_Teste 67` com $\hat{p} = 0{,}1128$), enquanto as outras $65\text{ amostras}$ convergiram para $\hat{p} \ge 0{,}99$ ou $\hat{p} \le 0{,}005$.

---

### 5.3. Decomposição de Latência e Perfilamento de Telemetria de Hardware
A Tabela 3 consolida as estatísticas descritivas das variáveis geométricas, temporais e de utilização de recursos computacionais coletadas ao longo das $70\text{ inferências}$ em lote.

**Tabela 3 — Estatísticas Descritivas Globais de Latência, Geometria e Recursos de Sistema ($N = 70$)**

| Variável Monitorada | Média ($\mu$) | Desvio Padrão ($\sigma$) | Mínimo | Quartil 25% ($Q_1$) | Mediana ($Q_2$) | Quartil 75% ($Q_3$) | Máximo | Participação no Ciclo |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Latência Estágio 1 — OpenCV ($\text{ms}$)** | $2{,}98$ | $4{,}27$ | $0{,}89$ | $1{,}82$ | $2{,}41$ | $3{,}05$ | $36{,}74$ | $0{,}03\%$ |
| **Latência Estágio 2 — EfficientNet ($\text{ms}$)** | $156{,}63$ | $38{,}95$ | $121{,}56$ | $142{,}10$ | $149{,}20$ | $156{,}80$ | $414{,}53$ | $1{,}72\%$ |
| **Latência Estágio 3 — LLaMA-3-8B ($\text{ms}$)** | $8.946{,}49$ | $1.202{,}87$ | $4.485{,}08$ | $8.592{,}60$ | $9.064{,}04$ | $9.495{,}80$ | $11.529{,}21$ | $98{,}25\%$ |
| **Latência Total do Pipeline ($\text{ms}$)** | $9.106{,}19$ | $1.219{,}57$ | $4.639{,}22$ | $8.756{,}50$ | $9.210{,}86$ | $9.646{,}60$ | $11.802{,}77$ | $100{,}00\%$ |
| **Contagem de Bolhas — OpenCV ($N_{\text{bolhas}}$)** | $232{,}27$ | $156{,}35$ | $7{,}00$ | $88{,}25$ | $219{,}00$ | $367{,}50$ | $566{,}00$ | — |
| **Fração de Área de Vapor — OpenCV ($\%$)** | $4{,}39\%$ | $3{,}36\%$ | $0{,}03\%$ | $1{,}42\%$ | $3{,}88\%$ | $6{,}85\%$ | $12{,}07\%$ | — |
| **Utilização de Processador — CPU ($\%$)** | $55{,}47\%$ | $6{,}03\%$ | $52{,}00\%$ | $52{,}50\%$ | $53{,}10\%$ | $54{,}80\%$ | $82{,}80\%$ | — |
| **Ocupação de Memória — RAM ($\%$)** | $87{,}13\%$ | $2{,}14\%$ | $83{,}30\%$ | $85{,}60\%$ | $87{,}60\%$ | $88{,}50\%$ | $92{,}70\%$ | — |

A Figura 3 apresenta a evolução temporal da latência média agrupada em blocos sequenciais de $5\text{ testes}$, correlacionada aos picos de utilização de CPU e memória RAM.

![alt text](/docs/pt_br/pipeline_03/images/figura03_pipeline03.jpg)

#### Diagnóstico do Fenômeno de Saturação de Memória (Testes 38 a 46):
A inspeção cruzada das curvas de telemetria explica com exatidão física o comportamento de cauda longa observado nos dados:
1. **Erradicação do *Cold Start*:** Graças ao método `warmup()`, o Teste 1 iniciou em $9.264{,}38\text{ ms}$, eliminando completamente o salto de $23.835{,}54\text{ ms}$ registrado na Iteração 2.
2. **Estabilidade Nominal (Testes 1 a 37 e 47 a 70):** Em regime estacionário, o Ollama aloca $6\text{ threads}$ correspondentes aos núcleos principais do Core i5-1335U, mantendo o uso de CPU rigorosamente cravado entre $52{,}0\%$ e $56{,}0\%$ e a latência da EfficientNet-B0 em $\sim 149\text{ ms}$.
3. **Saturação de RAM e *Memory Swapping* (Testes 38 a 46):** A partir do Teste 38 (`ID_Teste 37`), a ocupação da memória RAM de $15{,}7\text{ GB}$ ultrapassou o limiar crítico de $90{,}0\%$, atingindo $92{,}7\%$ no Teste 44 (`ID_Teste 43`). Nesse instante, o gerenciador de memória virtual do Windows acionou a paginação em disco SSD (*pagefile.sys*), elevando o uso de CPU para $82{,}8\%$. Como consequência direta da contenção de barramento de memória e cache L3:
   * A latência do LLaMA-3-8B atingiu o pico global de $11.529{,}21\text{ ms}$ no Teste 44 (`ID_Teste 43`);
   * A latência da EfficientNet-B0 (normalmente $\sim 149\text{ ms}$) saltou para $414{,}53\text{ ms}$ no Teste 45 (`ID_Teste 44`);
   * A latência do OpenCV (normalmente $\sim 2{,}4\text{ ms}$) atingiu seu máximo de $36{,}74\text{ ms}$ no mesmo período.
   Após a liberação de páginas de memória pelo sistema operacional no Teste 47, a RAM recuou para $\sim 86\%$ e as latências de todos os três estágios retornaram imediatamente ao patamar nominal.

---

### 5.4. Análise da Falha Geométrica do OpenCV e Assimetria Cognitiva do LLaMA-3-8B
Ao estratificar os resultados do Estágio 1 (OpenCV) e do Estágio 3 (LLaMA-3-8B) em função da classe preditiva, emergem duas constatações fundamentais sobre a dinâmica do Pipeline 03 (Tabela 4).

**Tabela 4 — Comparação Estratificada entre os Regimes de Incubação e Expulsão no Pipeline 03**

| Indicador Analítico | Regime de Incubação ($N_{\text{pred}} = 33$) | Regime de Expulsão ($N_{\text{pred}} = 37$) | Diferença Relativa ($\Delta\%$) | Interpretação Físico-Computacional |
| :--- | :---: | :---: | :---: | :--- |
| **Contagem Média de Bolhas ($N_{\text{bolhas}}$)** | $152{,}45\text{ bolhas}$ | $303{,}46\text{ bolhas}$ | $+99{,}05\%$ | Maior fragmentação de bordas e turbulência óptica na expulsão. |
| **Área Média de Vapor ($\alpha_{\text{área}}$)** | $2{,}81\%$ | $5{,}80\%$ | $+106{,}41\%$ | Dobro de área de contorno, porém **subestimada em ordem de grandeza**. |
| **Área Máxima Registrada ($\max \alpha_{\text{área}}$)** | $6{,}82\%$ | $12{,}07\%$ | $+76{,}98\%$ | Nenhuma amostra atingiu o limiar físico esperado de $\ge 20\%$ na expulsão. |
| **Latência Média do LLaMA-3 ($\text{ms}$)** | $8.512{,}40\text{ ms}$ | $9.333{,}65\text{ ms}$ | $+9{,}65\%$ | Respostas de alerta com discrepância de sensor exigem mais tokens que avisos normais. |
| **Latência Mínima Total ($\text{ms}$)** | $4.639{,}22\text{ ms}$ (`ID 21`) | $8.637{,}98\text{ ms}$ (`ID 11`) | $+86{,}19\%$ | Frases curtas geradas para estabilidade em *Incubação* encerram o loop antes de $45\text{ tokens}$. |

#### 1. A Falha Topológica do Filtro de Canny no Escoamento Bifásico:
Nos experimentos físicos do CCN, o regime de *Expulsão* caracteriza-se por grandes colunas de vapor que ocupam entre $30\%$ e $80\%$ da seção transversal do tubo (conforme corroborado pelo Pipeline 04 com YOLOv8-seg, onde o máximo atingiu $80{,}55\%$). Todavia, no Pipeline 03, a área máxima reportada pelo OpenCV em toda a bateria de $70\text{ imagens}$ foi de apenas **$12{,}07\%$** (`ID_Teste 10`), com média de $5{,}80\%$ nos casos de expulsão.

Isso ocorre porque as grandes bolsas de vapor (*slugs*) interceptam as bordas laterais da janela de recorte ($224 \times 224$) ou apresentam gradientes suaves de refração interna, fazendo com que o filtro de Canny gere **contornos abertos** (`cv2.contourArea` retorna área próxima de zero para linhas abertas de $1\text{ pixel}$ de espessura), enquanto a turbulência do líquido ao redor gera centenas de micro-contornos fechados de reflexo luminoso (até $566\text{ contornos}$ por frame).

#### 2. O Fenômeno de Propagação de Erro em Cascata no LLaMA-3-8B:
O *prompt* injetado no Estágio 3 continha a regra de segurança: *"If 'Expulsão crítica' and vapor < 20%, warn about sensor discrepancy. Give 1 short operational command sentence."*

Como a `EfficientNet-B0` (Estágio 2) acertou $100\%$ das expulsões reais, mas o `OpenCV` (Estágio 1) entregou $\alpha_{\text{área}} < 12{,}07\% < 20\%$ em absolutamente todas as imagens, o LLaMA-3-8B identificou o conflito matemático em **$100\%$ dos $37\text{ casos}$ preditos como Expulsão**, gerando comandos padronizados de alerta de falha de instrumentação, tais como:
> *"Verify sensor calibration immediately and initiate emergency cooling protocol due to critical expulsion state conflicting with low 5.42% vapor reading."*

Por outro lado, nos $33\text{ casos}$ preditos como *Incubação*, nos quais não havia conflito lógico no *prompt*, o LLaMA-3-8B emitiu diretrizes diretas de manutenção operacional (*"Maintain normal operation and continue monitoring bubble nucleation rate"*), gerando menos tokens e explicando o menor tempo de resposta observado no Teste 22 (`ID_Teste 21`, latência total de $4.639{,}22\text{ ms}$).

---

## 6. Discussão Científica e Contribuições para a Tese de Doutorado

Os resultados empíricos obtidos na execução do Pipeline 03 respondem diretamente às questões norteadoras do Plano de Trabalho de Doutorado do IPEN/CNEN:

1. **Raciocínio Lógico vs. Cegueira Perceptual do LLM Puro:** Comprovou-se que um LLM puramente textual de $8\text{B}$ parâmetros (LLaMA-3) possui excelente aderência lógica às restrições de engenharia fornecidas via *prompt*, sem apresentar alucinações sintáticas sob temperatura $\tau = 0{,}1$. Entretanto, por estar privado do tensor visual da imagem, o LLaMA-3 torna-se **refém da qualidade dos extratores numéricos precedentes**. Quando o OpenCV subestima a fração de vazio por falha de fechamento de contorno, o LLM não tem como inspecionar a imagem para perceber que o tubo está tomado por vapor, caracterizando a **Propagação de Erro em Cascata**.
2. **Superioridade do *Transfer Learning* com Escala Corrigida em *Low-Data Regime*:** A `EfficientNet-B0`, quando devidamente ajustada com reescalonamento $[0, 255]$ e congelamento de `BatchNormalization`, demonstrou capacidade extraordinária de generalização, alcançando $97{,}14\%$ de acurácia e $\text{MCC} = 0{,}9444$ em $70\text{ imagens}$ de teste tendo visto apenas $30\text{ imagens}$ no treinamento. Isso supera os $90{,}00\%$ da CNN customizada (Pipeline 01) e os $95{,}00\%$ do Pipeline 04.
3. **Viabilidade em Hardware de Baixo Custo e *Trade-off* Temporal:** A arquitetura desacoplada com Ollama viabilizou a execução local de um modelo de $8\text{ bilhões}$ de parâmetros em um notebook convencional sem GPU dedicada. Contudo, a latência média de $9{,}11\text{ s}$ ($0{,}11\text{ FPS}$) restringe o Estágio 3 a **funções de supervisão assíncrona, auditoria e emissão de relatórios (XAI)**, enquanto o intertravamento rápido de segurança da bancada deve ser acionado diretamente pela saída do Estágio 2 (`EfficientNet-B0`, que opera em $156{,}63\text{ ms}$ ou $\sim 6{,}4\text{ FPS}$ mesmo em CPU).

---

## 7. Conclusão

O desenvolvimento e a validação experimental do **Pipeline 03 (OpenCV + EfficientNet-B0 + LLaMA-3-8B)** cumpriram integralmente os objetivos propostos para a Onda 1 do cronograma de pesquisa. A refatoração arquitetural consolidada na classe `UnifiedPipeline03` solucionou os problemas de inicialização a frio (*Cold Start*), eliminou o colapso de representação na rede convolucional e implementou o monitoramento contínuo de telemetria de hardware.

Conclui-se que:
* A **EfficientNet-B0** (Estágio 2) deve ser mantida como um dos classificadores mais precisos e eficientes do projeto ($97{,}14\%$ de acurácia com apenas $30\text{ amostras}$ de treino);
* O algoritmo clássico de bordas do **OpenCV** (Estágio 1) é inadequado para quantificar a fração de vazio real em escoamentos bifásicos turbulentos, devendo ser substituído por segmentação semântica/instâncias baseada em aprendizado profundo (YOLOv8-seg ou SAM, conforme previsto nos Pipelines 06, 08, 10 e 11);
* O **LLaMA-3-8B** (Estágio 3) atua com elevada precisão lógica na síntese de pareceres operacionais explicáveis, mas requer que o Estágio 1 forneça áreas geométricas fidedignas ou que o Estágio 3 seja substituído por um modelo multimodal capaz de enxergar a imagem diretamente (como o LLaVA-1.5 no Pipeline 08) para dirimir conflitos de sensores.

---

## Referências Bibliográficas (Padrão ABNT NBR 6023:2018)

ALAYRAC, Jean-Baptiste et al. Flamingo: a visual language model for few-shot learning. **Advances in Neural Information Processing Systems (NeurIPS)**, v. 35, p. 23716-23736, 2022. Disponível em: https://arxiv.org/abs/2204.14198. Acesso em: 15 nov. 2026.

BOMMASANI, Rishi et al. On the opportunities and risks of foundation models. **arXiv preprint arXiv:2108.07258**, Stanford University, p. 1-214, 2021. Disponível em: https://arxiv.org/abs/2108.07258. Acesso em: 15 nov. 2026.

BRADSKI, Gary. The OpenCV Library. **Dr. Dobb's Journal of Software Tools**, v. 25, n. 11, p. 120-125, 2000.

CANNY, John. A computational approach to edge detection. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, v. PAMI-8, n. 6, p. 679-698, nov. 1986. DOI: https://doi.org/10.1109/TPAMI.1986.4767851.

DETTMERS, Tim; PAGNONI, Artidoro; HOLTZMAN, Ari; ZETTLEMOYER, Luke. QLoRA: efficient finetuning of quantized LLMs. **Advances in Neural Information Processing Systems (NeurIPS)**, v. 36, p. 10088-10115, 2023. Disponível em: https://arxiv.org/abs/2305.14314. Acesso em: 15 nov. 2026.

DIAS, L. S. et al. Classification of two-phase flow pattern in microtubes by image processing and machine learning. **International Journal of Multiphase Flow**, v. 185, p. 105122, 2025. DOI: https://doi.org/10.1016/j.ijmultiphaseflow.2024.105122.

DUBEY, Abhimanyu et al. (AI@Meta). The Llama 3 herd of models. **arXiv preprint arXiv:2407.21783**, p. 1-92, jul. 2024. Disponível em: https://arxiv.org/abs/2407.21783. Acesso em: 15 nov. 2026.

HUA, Jinwei; DONG, Yi; DING, Zhengtao; HUANG, Xiaowei. Enhancing robustness of LLM-driven multi-agent systems through randomized smoothing. **Chinese Journal of Aeronautics**, 2024.

IOFFE, Sergey; SZEGEDY, Christian. Batch normalization: accelerating deep network training by reducing internal covariate shift. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML), 32., 2015, Lille. **Proceedings [...]**. Lille: PMLR, v. 37, p. 448-456, 2015.

ISHII, Mamoru; HIBIKI, Takashi. **Thermo-fluid dynamics of two-phase flow**. 2. ed. New York: Springer, 2011. DOI: https://doi.org/10.1007/978-1-4419-7985-8.

JIANG, Peiyuan et al. A review of Yolo algorithm developments. **Procedia Computer Science**, v. 199, p. 1066-1073, 2022. DOI: https://doi.org/10.1016/j.procs.2022.01.135.

KINGMA, Diederik P.; BA, Jimmy. Adam: a method for stochastic optimization. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS (ICLR), 3., 2015, San Diego. **Proceedings [...]**. San Diego, 2015. Disponível em: https://arxiv.org/abs/1412.6980. Acesso em: 15 nov. 2026.

LIN, Ji et al. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. **Proceedings of Machine Learning and Systems (MLSys)**, v. 6, p. 87-100, 2024. Disponível em: https://arxiv.org/abs/2306.09782. Acesso em: 15 nov. 2026.

LUO, Y.; ZHANG, T.; CHEN, Z.; XU, J. Prediction of two-phase flow patterns based on machine learning. **International Journal of Multiphase Flow**, v. 165, p. 104412, 2023.

MADRY, Aleksander et al. Towards deep learning models resistant to adversarial attacks. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS (ICLR), 6., 2018, Vancouver. **Proceedings [...]**. Vancouver, 2018. Disponível em: https://arxiv.org/abs/1706.06083. Acesso em: 15 nov. 2026.

MASOTTI, Paulo Henrique Ferraz; DE MESQUITA, Roberto Navarro. Análise de circuitos de circulação natural em instalações nucleares. **Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN)**, São Paulo, 2015.

QIAO, W. et al. Real-time detection of slug flow in subsea pipelines by embedding a Yolo object detection algorithm into Jetson Nano. **Journal of Marine Science and Engineering**, v. 11, n. 9, p. 1658, 2023. DOI: https://doi.org/10.3390/jmse11091658.

QUINTINO, A. M.; STANKO, M. Machine learning uncertainty framework applied to gas-liquid horizontal pipe flow. **International Journal of Multiphase Flow**, v. 187, p. 105184, 2025. DOI: https://doi.org/10.1016/j.ijmultiphaseflow.2025.105184.

RADFORD, Alec et al. Learning transferable visual models from natural language supervision. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML), 38., 2021. **Proceedings [...]**. PMLR, v. 139, p. 8748-8763, 2021.

SILVA, Marcones Cleber Brito da. **Robustez e generalização de modelos fundamentais de visão e linguagem em escoamento bifásico**. 2025. 22 f. Plano de Trabalho de Doutorado — Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN), Universidade de São Paulo, São Paulo, 2025.

TAN, Mingxing; LE, Quoc V. EfficientNet: rethinking model scaling for convolutional neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML), 36., 2019, Long Beach. **Proceedings [...]**. Long Beach: PMLR, v. 97, p. 6105-6114, 2019. Disponível em: https://arxiv.org/abs/1905.11946. Acesso em: 15 nov. 2026.

TORRES, W. M.; TORRES, L. S. **Experimentos de termoidráulica em circuito de circulação natural**. São Paulo: Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN), 2006.

YANG, Kai et al. Foundation models for scientific applications. **Nature Machine Intelligence**, v. 4, p. 489-502, 2022. DOI: https://doi.org/10.1038/s42256-022-00590-2.

ZHANG, Y.; WANG, H.; LI, X. Long short-term memory networks for predicting multiphase flow dynamics. **Journal of Fluid Mechanics**, v. 950, p. 1-25, 2023.