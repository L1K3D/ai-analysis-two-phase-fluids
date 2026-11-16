# Manual Oficial de Padronização, Arquitetura e Desenvolvimento dos Pipelines (1 a 12)

**Projeto de Pesquisa:** Robustez e Generalização de Modelos Fundamentais de Visão e Linguagem em Escoamento Bifásico  
**Instituição:** Instituto de Pesquisas Energéticas e Nucleares (IPEN/CNEN) — Centro de Engenharia Nuclear (CEN)  
**Repositório:** `ai-analysis-two-phase-fluids`  
**Equipe Técnica:** Enzo, Erikson, Heitor, Paulo e William  

---

## 1. Visão Geral e Fluxo Obrigatório de Entrega

Bem-vindos ao guia oficial de engenharia e documentação científica do projeto de análise termo-hidráulica de escoamentos bifásicos no Circuito de Circulação Natural (CCN). Para que os resultados dos **12 Pipelines Híbridos** sejam matematicamente comparáveis na tese de doutorado e nos artigos científicos derivados, **todos os integrantes devem seguir rigorosamente a mesma metodologia, estrutura de pastas, divisão de dados e padrão documental**.

Cada pipeline atribuído na planilha de cronograma deve cumprir obrigatoriamente uma esteira de **4 Etapas Sequenciais**:

```
+-----------------------------------------------------------------------------------+
| ETAPA 1: Relatório Técnico de Compreensão                                         |
| -> Estudo teórico dos 3 estágios, equacionamento e plano de ataque conceitual     |
| -> Entregáveis: relatorio_compreensao_pipelineXX.md e .pdf                        |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
| ETAPA 2: Construção e Treinamento do Modelo / Pipeline Unificado                  |
| -> Divisão 30% Treino (30 imgs) / 70% Teste (70 imgs salvas em cache .npz)        |
| -> Classe UnifiedPipelineXX com Warm-up e telemetria de hardware (CPU/RAM/Tempo)  |
| -> Entregáveis: scripts .py, pesos treinados (.keras/.pth) e cache (.npz)         |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
| ETAPA 3: Execução da Bateria de Testes e Coleta de Benchmark                      |
| -> Execução sobre as 70 imagens de teste com medição isolada por estágio          |
| -> Entregável: benchmarks/benchmark_pipeline_XX.csv                               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------+-----------------------------------------+
| ETAPA 4: Relatório Final (Artigo Científico Experimental)                         |
| -> Confronto Expectativa vs. Realidade, evolução iterativa, cálculos (F1, MCC),   |
|    gráficos de alta resolução salvos em /images e referências em formato ABNT     |
| -> Entregáveis: relatorio_final_pipelineXX.md, .pdf e imagens .jpg/.png           |
+-----------------------------------------------------------------------------------+
```

---

## 2. Configuração Inicial do Ambiente e Git Workflow (Passo a Passo)

### Passo 2.1: Clonar o Repositório
1. Abra o terminal do seu sistema (PowerShell, CMD, Git Bash ou o próprio terminal integrado do VS Code).
2. Navegue até a pasta onde você guarda seus projetos (ex.: `C:\projects\`) e execute o comando de clonagem:
   ```bash
   git clone <URL_DO_REPOSITORIO_GITHUB>
   cd ai-analysis-two-phase-fluids
   ```

### Passo 2.2: Criar a sua Branch a partir da `develop`
Nenhum integrante deve programar diretamente na `main` ou na `develop`. Todo o trabalho deve ser feito em uma branch própria derivada da `develop`.

1. No terminal, dentro da pasta do projeto, mude para a branch `develop` e garanta que ela está atualizada:
   ```bash
   git checkout develop
   git pull origin develop
   ```
2. Crie e mude para a sua branch de desenvolvimento seguindo **exatamente** a nomenclatura `feature-develop-<nickname_do_github>`:
   ```bash
   git checkout -b feature-develop-seu_nickname_do_github
   ```
   *(Exemplo: se o seu usuário no GitHub for `enzosilva`, o comando será `git checkout -b feature-develop-enzosilva`).*
3. Confirme se você está na branch correta digitando:
   ```bash
   git branch
   ```
   *(O asterisco `*` deve estar ao lado de `feature-develop-<seu_nickname>`).*

### Passo 2.3: Inserir a Pasta `configs_scripts` (Baixada do WhatsApp)
Por questões de segurança da informação, as credenciais de autenticação da conta de serviço do Google Drive e os IDs das pastas de imagens não sobem para o GitHub.

1. Vá no grupo do **WhatsApp** do projeto e baixe o arquivo compactado referente à pasta **`configs_scripts`**.
2. Extraia o conteúdo e cole a pasta `configs_scripts` **diretamente na raiz do projeto clonado** (`ai-analysis-two-phase-fluids/configs_scripts/`).
3. Verifique se dentro de `configs_scripts/` estão presentes os arquivos:
   * `auth_google_drive_config.json` (Chave privada da Service Account da Google Cloud);
   * `general_infos.env` (Variáveis de ambiente `SCOPES`, `EXPULSION_FOLDER_ID` e `INCUBATION_FOLDER_ID`);
   * `project-notes.txt`.

> ### ⚠️ REGRA DE OURO: PROIBIDO ALTERAR O `.gitignore`!
> **NÃO MEXA EM ABSOLUTAMENTE NADA NO ARQUIVO `.gitignore` DA RAIZ DO PROJETO!**  
> O arquivo `.gitignore` foi configurado cirurgicamente para impedir que as chaves privadas da pasta `configs_scripts` vazem na internet (o que derrubaria a credencial do Google Drive de todo o grupo) e para gerenciar quais artefatos pesados podem ou não subir para o repositório. Qualquer modificação no `.gitignore` terá o Pull Request bloqueado.

---

## 3. Estrutura Oficial de Pastas e Onde Salvar Cada Arquivo

Tomando como referência a implementação consolidada do **Pipeline 03**, a árvore de diretórios do seu VS Code deve ficar rigorosamente organizada conforme o esquema abaixo (substitua `XX` pelo número do seu pipeline com dois dígitos, ex.: `01`, `02`, `05`, `08`):

```
ai-analysis-two-phase-fluids/
├── .vscode/
├── benchmarks/
│   └── benchmark_pipeline_XX.csv                  <-- [OBRIGATÓRIO] CSV final gerado no teste (70 amostras)
├── configs_scripts/                               <-- [BAIXAR DO WHATSAPP] Credenciais do Google Drive
│   ├── auth_google_drive_config.json
│   ├── general_infos.env
│   └── project-notes.txt
├── docs/
│   ├── pipeline_XX/                               <-- Pasta de documentação exclusiva do seu Pipeline
│   │   ├── images/                                <-- Gráficos gerados para o artigo final
│   │   │   ├── figura01_pipelineXX.jpg
│   │   │   ├── figura02_pipelineXX.jpg
│   │   │   └── figura03_pipelineXX.jpg
│   │   ├── relatorio_compreensao_pipelineXX.md    <-- Artigo de Compreensão em Markdown (Etapa 1)
│   │   ├── relatorio_compreensao_pipelineXX.pdf   <-- Artigo de Compreensão exportado em PDF
│   │   ├── relatorio_final_pipelineXX.md          <-- Artigo Final Experimental em Markdown (Etapa 4)
│   │   └── relatorio_final_pipelineXX.pdf         <-- Artigo Final Experimental exportado em PDF
│   └── utils_references/                          <-- PDFs de apoio (Plano de Doutorado, artigos base)
├── python_scripts/
│   ├── pipeline_XX/                               <-- Pasta de códigos e modelos do seu Pipeline
│   │   ├── dados_teste_pX.npz                     <-- Cache das 70 imagens de teste (gerado no treino)
│   │   ├── <modelo>_tranning.py                   <-- Script de fine-tuning do classificador (Etapa 2)
│   │   ├── <modelo>_finetuned.keras (ou .pth)     <-- Pesos salvos do modelo ajustado
│   │   └── pipeline_XX.py                         <-- Script unificado de execução e telemetria (Etapa 3)
│   └── yolov8n-seg.pt                             <-- Pesos base do YOLO (quando aplicável)
├── utils/                                         <-- Funções compartilhadas (NÃO ALTERAR SEM AVISAR)
│   ├── utls_google_drive_auth.py
│   ├── utls_ia_models.py
│   ├── utls_install_libs.py
│   └── utls_work_images.py
├── .gitignore                                     <-- [PROIBIDO ALTERAR]
└── README.md
```

---

## 4. Passo a Passo das 4 Etapas de Construção (Exemplo Prático do Pipeline 03)

Abaixo está o roteiro detalhado de como executar cada uma das 4 etapas do seu pipeline, acompanhado dos **prompts de IA** que você pode copiar, adaptar com os nomes das tecnologias do seu pipeline e usar para acelerar sua produção mantendo o padrão científico do grupo.

---

### ETAPA 1: Relatório Técnico de Compreensão (`relatorio_compreensao_pipelineXX.md`)

Antes de escrever qualquer linha de código, você deve produzir um artigo técnico preliminar (de até 5 páginas) demonstrando domínio teórico sobre os 3 estágios do seu pipeline e como eles se conectam ao Plano de Trabalho de Doutorado do IPEN.

#### O que o Relatório de Compreensão DEVE conter:
1. **Cabeçalho e Resumo Acadêmico:** Identificando o projeto do IPEN/CNEN, o Circuito de Circulação Natural (CCN) e os 3 modelos do seu pipeline.
2. **Introdução e Contextualização:** Explicação física dos regimes de escoamento bifásico (*Incubação* vs. *Expulsão*) e o papel do seu pipeline na matriz de 12 modelos.
3. **Análise Técnica dos 3 Estágios (Com Diagrama em Bloco ASCII):**
   * *Estágio 1 (Contagem e Área):* Explicação matemática do algoritmo (OpenCV, YOLOv8-seg ou SAM) e da equação da fração de vazio planar ($\alpha_{\text{área}}$).
   * *Estágio 2 (Classificador de Estado):* Arquitetura da rede convolucional (CNN Customizada, ResNet-18 ou EfficientNet-B0), número de parâmetros e equação de perda.
   * *Estágio 3 (Motor de Ação):* Funcionamento do decisor (Regras Estritas `if/else`, Moondream2, LLaMA-3-8B ou LLaVA-1.5) e estratégia de quantização/execução local.
4. **Conceito da Aplicação e Engenharia de Prompt / Regras:** Exemplo exato do *prompt* ou da árvore lógica que será alimentada no Estágio 3.
5. **Plano de Ataque Conceitual e Protocolo de Teste:** Esquema sequencial de como o teste será conduzido e quais métricas serão medidas (Acurácia, $F_1$, Latência fim-a-fim $\Delta t_{\text{total}}$, uso de memória).
6. **Resultados Esperados:** Tabela estimando latência, consumo de RAM/VRAM, explicabilidade (XAI) e riscos (como propagação de erro em cascata).
7. **Referências Bibliográficas:** Artigos seminais dos modelos utilizados e do IPEN.

#### 🤖 Prompt de IA para ajudar na Etapa 1:
> *"Atue como pesquisador sênior em engenharia nuclear e aprendizado profundo.Estou desenvolvendo a Etapa 1 (Relatório de Compreensão) do **Pipeline XX** do projeto de doutorado do IPEN/CNEN sobre análise de escoamento bifásico em um Circuito de Circulação Natural (CCN). O meu pipeline utiliza:*  
> *- Estágio 1 (Contagem e Área de Bolhas): [INSERIR MODELO 1, ex: YOLOv8-seg]*  
> *- Estágio 2 (Classificador de Estado - Incubação vs Expulsão): [INSERIR MODELO 2, ex: ResNet-18]*  
> *- Estágio 3 (Motor de Ação Operacional): [INSERIR MODELO 3, ex: Moondream2]*  
> *- Objetivo do Teste: [COPIAR DA TABELA DOS 12 PIPELINES]*  
> *Escreva um artigo técnico completo em formato Markdown (`.md`) de até 5 páginas contendo: Resumo, Introdução contextualizada ao CCN do IPEN, Análise Técnica e Matemática detalhada dos 3 estágios (com diagramas ASCII e equações em LaTeX), Conceito da Aplicação com template de prompt/regras, Plano de Ataque Conceitual esquematizado, Tabela de Resultados Esperados (latência e memória) e Referências Bibliográficas acadêmicas reais."*

---

### ETAPA 2: Construção e Treinamento do Modelo (`python_scripts/pipeline_XX/`)

Durante o desenvolvimento do **Pipeline 03**, descobrimos armadilhas críticas de implementação que derrubavam a acurácia para $40\%$ ou travavam a máquina. Todos devem aplicar as **4 Soluções Padrão** abaixo nos seus scripts:

#### Lições Técnicas Obrigatórias (Aprendidas no Pipeline 03):
1. **Partição Invertida de Dados (30% Treino / 70% Teste) + Cache `.npz`:**
   * Nossas pastas do Google Drive fornecem $50\text{ imagens}$ de `expulsion` (rótulo `0`) e $50\text{ imagens}$ de `incubation` (rótulo `1`).
   * Para testar a **Capacidade de Generalização sob Escassez de Dados** exigida no doutorado, usamos **`test_size=0.70`** com `stratify=y_completo` e `random_state=42`. Isso separa **30 imagens para treinar** o classificador e salva **70 imagens inéditas para o teste** em um arquivo local `dados_teste_pX.npz`.
   * Assim, o script de benchmark carrega as 70 imagens do SSD em menos de 1 segundo e garante que **nenhuma imagem vista no treino seja usada no teste**.
2. **Atenção à Escala de Pixels `[0, 1]` vs. `[0, 255]`:**
   * A nossa função `build_dataset()` em `utils/utls_work_images.py` já divide os pixels por `255.0` (entregando matrizes `float32` entre `0.0` e `1.0`).
   * **Cuidado:** Redes como a `EfficientNetB0` do Keras possuem uma camada interna que divide por `255` novamente! Se você passar a matriz em `[0, 1]` direto para ela, a imagem fica toda preta e a rede chuta uma classe só ($50\% - 60\%$ de acurácia viciada). No Pipeline 03, resolvemos isso colocando uma camada `Rescaling(255.0)` na entrada do modelo e mantendo as camadas `BatchNormalization` com `trainable = False` durante o *fine-tuning*. Com isso, a acurácia saltou de $40\%$ para **$97{,}14\%$**!
3. **Execução de LLMs e VLMs Locais via Ollama (Sem Estourar RAM/VRAM):**
   * Para os pipelines que usam **LLaMA-3-8B**, **Moondream2** ou **LLaVA-1.5** no Estágio 3, **não importe** `AutoModelForCausalLM` do Hugging Face dentro do script principal (isso gera erro de *circular import* do PyTorch no Python 3.13 do Windows e estoura a memória).
   * Instale o **[Ollama](https://ollama.com/)** para Windows e baixe o modelo do seu pipeline no terminal:
     * Para LLaMA-3-8B: `ollama pull llama3`
     * Para Moondream2: `ollama pull moondream`
     * Para LLaVA-1.5: `ollama pull llava`
   * O seu script Python se comunicará com o Ollama via requisição HTTP local (`http://localhost:11434/api/generate`) usando `requests.Session()`.
4. **Arquitetura Orientada a Objetos (`UnifiedPipelineXX`) com `warmup()` e Telemetria (`psutil`):**
   * Encapsule os 3 estágios em uma classe `UnifiedPipelineXX`.
   * Crie um método `warmup()` que passa uma imagem vazia pelos modelos antes de iniciar o loop das 70 imagens. No Pipeline 03, isso eliminou o *Cold Start* de $23{,}8\text{ segundos}$ da primeira imagem!
   * Meça a latência individual de cada estágio (`Lat_Estagio1_ms`, `Lat_Estagio2_ms`, `Lat_Estagio3_ms`), a latência total (`Latencia_ms`), o uso de CPU (`CPU_Uso_%`), o uso de RAM (`RAM_Uso_%`) e as especificações da máquina (`Hardware_Specs`).

#### Código-Base de Referência 1: Script de Treinamento (`python_scripts/pipeline_03/efficient_net_tranning.py`)
Use este modelo como base para criar o script de treinamento do seu classificador do Estágio 2:

```python
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Rescaling, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Localiza a raiz do projeto automaticamente
SCRIPT_DIR = Path(__file__).resolve().parent
project_root = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
     if (parent / "utils" / "utls_install_libs.py").is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Não foi possível localizar utils/utls_install_libs.py")
sys.path.insert(0, str(project_root))

from utils.utls_install_libs import install_missing_libs
from utils.utls_work_images import build_dataset

install_missing_libs(['tensorflow', 'pandas', 'scikit-learn'])

LIMITE_POR_CLASSE = 50 
print(f"📥 [1/4] Baixando imagens do Google Drive (Limite por classe: {LIMITE_POR_CLASSE})...")
X_exp, y_exp = build_dataset(limit_per_class_collected=LIMITE_POR_CLASSE, folder_name_collected="expulsion")
X_inc, y_inc = build_dataset(limit_per_class_collected=LIMITE_POR_CLASSE, folder_name_collected="incubation")

X_completo = np.concatenate((X_exp, X_inc), axis=0)
y_completo = np.concatenate((y_exp, y_inc), axis=0)

# Partição Padrão do Grupo: 30% Treino (30 imgs) | 70% Teste (70 imgs)
X_train, X_test, y_train, y_test = train_test_split(
    X_completo, y_completo, test_size=0.70, stratify=y_completo, random_state=42
)

caminho_cache_teste = SCRIPT_DIR / "dados_teste_p3.npz"
np.savez_compressed(caminho_cache_teste, X_test=X_test, y_test=y_test)
print(f"📦 Lote dividido: {len(X_train)} imagens para Treino | {len(X_test)} imagens salvas para Teste.")

print("🧠 [2/4] Montando a arquitetura do Classificador...")
efficient_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

efficient_base.trainable = True
for layer in efficient_base.layers[:-20]:
    layer.trainable = False
for layer in efficient_base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

modelo = Sequential([
    Rescaling(255.0, input_shape=(224, 224, 3)),
    efficient_base,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # 0 = Expulsão, 1 = Incubação
])

modelo.compile(optimizer=Adam(learning_rate=3e-4), loss='binary_crossentropy', metrics=['accuracy'])

print("🔥 [3/4] Treinando no lote enxuto (12 épocas)...")
modelo.fit(X_train, y_train, epochs=12, batch_size=8, validation_data=(X_test[:20], y_test[:20]), verbose=1)

caminho_modelo = SCRIPT_DIR / "efficientnet_finetuned.keras"
modelo.save(caminho_modelo)
print(f"✅ [4/4] Modelo salvo em: {caminho_modelo}")
```

---

### ETAPA 3: Execução do Pipeline e Benchmark (`python_scripts/pipeline_XX/pipeline_XX.py`)

Com o modelo treinado e o cache `.npz` gerado, execute o script integrador que roda as **70 imagens de teste** e salva o resultado em **`benchmarks/benchmark_pipeline_XX.csv`**.

#### Código-Base de Referência 2: Script Unificado (`python_scripts/pipeline_03/pipeline_03.py`)
*(Observação: para VLMs como `moondream` e `llava` no Ollama, basta converter a matriz `img_norm` para Base64 e enviá-la no campo `"images": [img_b64]` do JSON do Ollama).*

```python
import os
import sys
import time
import platform
import psutil
import cv2
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

SCRIPT_DIR = Path(__file__).resolve().parent
project_root = next(
    (parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents)
     if (parent / "utils" / "utls_install_libs.py").is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Não foi possível localizar utils/utls_install_libs.py")
sys.path.insert(0, str(project_root))

from utils.utls_install_libs import install_missing_libs
from utils.utls_work_images import build_dataset

install_missing_libs(['requests', 'pandas', 'tensorflow', 'scikit-learn', 'opencv-python', 'psutil'])

class UnifiedPipeline03:
    def __init__(self, keras_model_path, ollama_model="llama3"):
        self.classifier = load_model(keras_model_path)
        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = ollama_model
        self.http_session = requests.Session()
        self.hw_specs = (
            f"CPU: {platform.processor()} ({psutil.cpu_count(logical=False)}C/{psutil.cpu_count(logical=True)}T) | "
            f"RAM: {round(psutil.virtual_memory().total / (1024**3), 1)}GB"
        )

    def warmup(self):
        print("🔥 Aquecendo motores (Warm-up)...")
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.classifier.predict(dummy_img, verbose=0)
        try:
            self.http_session.post(self.ollama_url, json={
                "model": self.ollama_model, "prompt": "Reply OK.", "stream": False,
                "keep_alive": "15m", "options": {"num_predict": 5}
            }, timeout=60)
        except Exception as e:
            print(f"⚠️ Aviso no Warm-up do Ollama: {e}")

    def _stage1_opencv(self, img_norm):
        t0 = time.perf_counter()
        img_uint8 = (img_norm * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        bordas = cv2.Canny(gray, 50, 150)
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        qtd_bolhas = len(contornos)
        pct_vapor = round((sum(cv2.contourArea(c) for c in contornos) / (224 * 224)) * 100, 2)
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return qtd_bolhas, pct_vapor, lat_ms

    def _stage2_efficientnet(self, img_norm):
        t0 = time.perf_counter()
        img_batch = np.expand_dims(img_norm, axis=0)
        prob = float(self.classifier.predict(img_batch, verbose=0)[0][0])
        classe = 1 if prob > 0.5 else 0
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return classe, round(prob, 4), lat_ms

    def _stage3_llama3(self, classe_ia, qtd, pct):
        t0 = time.perf_counter()
        estado = "Expulsão crítica" if classe_ia == 0 else "Incubação inicial"
        prompt = (
            f"Act as a nuclear safety engineer. Visual state: '{estado}'. "
            f"Sensor: {qtd} bubbles, {pct}% vapor area. "
            f"If 'Expulsão crítica' and vapor < 20%, warn about sensor discrepancy. "
            f"Give 1 short operational command sentence."
        )
        payload = {
            "model": self.ollama_model, "prompt": prompt, "stream": False,
            "keep_alive": "15m", "options": {"num_predict": 45, "temperature": 0.1}
        }
        try:
            resp = self.http_session.post(self.ollama_url, json=payload, timeout=90)
            resp.raise_for_status()
            acao = resp.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            acao = f"ERRO Ollama: {e}"
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return acao, lat_ms

    def run_single_image(self, img_norm):
        psutil.cpu_percent(interval=None)
        t_inicio = time.perf_counter()

        qtd, pct, lat_s1 = self._stage1_opencv(img_norm)
        classe, prob, lat_s2 = self._stage2_efficientnet(img_norm)
        acao, lat_s3 = self._stage3_llama3(classe, qtd, pct)

        lat_total = round((time.perf_counter() - t_inicio) * 1000, 2)
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent

        return {
            "Predicao_Classificador": "Incubação" if classe == 1 else "Expulsão",
            "Probabilidade_Classe_1": prob,
            "Qtd_Bolhas": qtd,
            "Vapor_%": pct,
            "Acao_Sugerida": acao,
            "Lat_Estagio1_ms": lat_s1,
            "Lat_Estagio2_ms": lat_s2,
            "Lat_Estagio3_ms": lat_s3,
            "Latencia_ms": lat_total,
            "CPU_Uso_%": cpu_pct,
            "RAM_Uso_%": ram_pct,
            "Hardware_Specs": self.hw_specs
        }

if __name__ == "__main__":
    caminho_cache = SCRIPT_DIR / "dados_teste_p3.npz"
    caminho_modelo = SCRIPT_DIR / "efficientnet_finetuned.keras"

    dados = np.load(caminho_cache)
    X_test, y_test = dados["X_test"], dados["y_test"]

    pipeline = UnifiedPipeline03(keras_model_path=caminho_modelo)
    pipeline.warmup()

    resultados = []
    for i in range(len(X_test)):
        metricas = pipeline.run_single_image(X_test[i])
        real_str = "Incubação" if y_test[i] == 1 else "Expulsão"
        resultados.append({"ID_Teste": i, "Real": real_str, **metricas})
        print(f"  -> [{i+1}/{len(X_test)}] Real: {real_str:9s} | Pred: {metricas['Predicao_Classificador']:9s} | Total: {metricas['Latencia_ms']} ms")

    df = pd.DataFrame(resultados)
    pasta_benchmarks = project_root / "benchmarks"
    pasta_benchmarks.mkdir(exist_ok=True)
    caminho_csv = pasta_benchmarks / "benchmark_pipeline_03.csv"
    df.to_csv(caminho_csv, index=False)
    print(f"\n✅ Benchmark salvo em: {caminho_csv}")
```

#### 🤖 Prompt de IA para ajudar nas Etapas 2 e 3:
> *"Preciso construir os scripts em Python para o **Pipeline XX** do meu projeto de escoamento bifásico, seguindo o padrão exato da classe `UnifiedPipeline03`. O meu pipeline usa:*  
> *- Estágio 1: [INSERIR ESTÁGIO 1, ex: YOLOv8n-seg]*  
> *- Estágio 2: [INSERIR ESTÁGIO 2, ex: ResNet-18 ou ResNet-50]*  
> *- Estágio 3: [INSERIR ESTÁGIO 3, ex: Moondream2 via API local do Ollama na porta 11434]*  
> *Gere dois scripts completos:*  
> *1. O script de treinamento do Estágio 2 importando `build_dataset` de `utils.utls_work_images`, dividindo 30% para treino (30 imagens) e 70% para teste (70 imagens), salvando o lote de teste em `dados_teste_pX.npz`, aplicando correção de escala `[0, 255]` se necessário e congelando camadas `BatchNormalization`.*  
> *2. O script unificado `pipeline_XX.py` que carrega o `.npz`, faz `warmup()`, mede a latência individual dos 3 estágios e a telemetria de CPU e RAM com `psutil`, e salva o resultado em `benchmarks/benchmark_pipeline_XX.csv`."*

---

### ETAPA 4: Construção do Relatório Final (`relatorio_final_pipelineXX.md`) e Gráficos

Após gerar o arquivo `benchmarks/benchmark_pipeline_XX.csv`, você deve produzir o **Artigo Científico Final de Análise Experimental** do seu pipeline.

#### O que o Relatório Final DEVE conter (Padrão Pipeline 03):
1. **Resumo e Palavras-Chave:** Síntese completa dos métodos, hardware utilizado, acurácia, $\text{MCC}$ e latências de cada estágio.
2. **Seção 1 — Introdução e Contextualização Científica:** Fundamentação termo-hidráulica no CCN do IPEN com citações no padrão ABNT (Autor, Ano).
3. **Seção 2 — Expectativa Teórica vs. Realidade de Implantação Operacional:** Tabela confrontando o que era esperado no Relatório de Compreensão (Etapa 1) com as especificações reais do seu computador (CPU, GPU, RAM) e os tempos reais medidos.
4. **Seção 3 — Metodologia Experimental e Evolução Iterativa:** Descrição de como o modelo foi ajustado, problemas enfrentados nos testes iniciais e como foram corrigidos até chegar na versão `UnifiedPipelineXX`.
5. **Seção 4 — Formulação Matemática dos 3 Estágios:** Dedução rigorosa em LaTeX das equações do Estágio 1, Estágio 2 e Estágio 3.
6. **Seção 5 — Resultados Experimentais e Análise Estatística Profunda:**
   * Tabela época a época do treinamento + **Figura 1** (Gráfico de decaimento da Loss);
   * **Figura 2** (Matriz de Confusão nas 70 imagens) + Cálculos explícitos passo a passo de Acurácia, Precisão, Recall, $F_1\text{-Score}$ para ambas as classes e Coeficiente de Correlação de Matthews ($\text{MCC}$);
   * Tabela de Estatísticas Descritivas Globais (Média, Desvio Padrão, Mínimo, $Q_1$, Mediana, $Q_3$, Máximo e $\%$ no ciclo para cada estágio, CPU e RAM) + **Figura 3** (Perfil temporal de latência e uso de hardware por bloco);
   * Análise Estratificada (*Incubação* vs. *Expulsão*) e diagnóstico de propagação de erro ou comportamento do Estágio 3.
7. **Seções 6 e 7 — Discussão Científica, Conclusão e Referências ABNT NBR 6023:2018.**

#### Como Gerar e Inserir as Figuras no seu `.md`:
1. Peça para a IA gerar as 3 figuras separadamente como imagens de alta resolução (ou gere via script `matplotlib`/`seaborn`).
2. Salve as imagens dentro de **`docs/pipeline_XX/images/`** com os nomes:
   * `figura01_pipelineXX.jpg`
   * `figura02_pipelineXX.jpg`
   * `figura03_pipelineXX.jpg`
3. No seu arquivo `relatorio_final_pipelineXX.md`, referencie as imagens usando o caminho relativo:
   ```markdown
   ![Figura 1 - Decaimento da Loss](./images/figura01_pipelineXX.jpg)
   ```

#### 🤖 Prompt de IA para ajudar na Etapa 4:
> *"Em anexo estão o meu `relatorio_compreensao_pipelineXX.md`, os logs de treinamento no terminal e o arquivo `benchmark_pipeline_XX.csv` com os resultados das 70 imagens de teste do **Pipeline XX**.*  
> *Desenvolva o Artigo Final de Análise Experimental (`relatorio_final_pipelineXX.md`) seguindo exatamente o padrão científico do Pipeline 03 do grupo. Não poupe detalhes, deduções matemáticas em LaTeX nem cálculos estatísticos. Inclua:*  
> *1. Tabela Expectativa vs. Realidade (com base no hardware registrado no CSV);*  
> *2. Metodologia iterativa realizada;*  
> *3. Formulação matemática completa dos 3 estágios;*  
> *4. Cálculos abertos de Acurácia, Precisão, Recall, F1-Score e MCC a partir da Matriz de Confusão;*  
> *5. Tabela estatística completa (Média, DP, Quartis, Mín/Máx) de latência por estágio, bolhas, vapor, CPU e RAM;*  
> *6. Análise estratificada entre Incubação e Expulsão e discussão para a tese de doutorado do IPEN;*  
> *7. Citações ao longo do texto e lista final de Referências Bibliográficas no padrão ABNT NBR 6023:2018.*  
> *Em seguida, gere separadamente as imagens de alta resolução da Figura 1 (Curva de Loss), Figura 2 (Matriz de Confusão) e Figura 3 (Perfil de Latência e Hardware por Bloco) para eu salvar na pasta `images/`."*

---

## 5. Como Converter os Arquivos Markdown (`.md`) para PDF no VS Code

Todos os relatórios (`relatorio_compreensao_pipelineXX.md` e `relatorio_final_pipelineXX.md`) devem estar salvos na pasta `docs/pipeline_XX/` tanto em formato **`.md`** quanto em formato **`.pdf`**.

Siga este passo a passo para instalar e usar a extensão **Markdown PDF** no VS Code:

1. **Instalar a Extensão:**
   * Com o VS Code aberto, clique no ícone de **Extensões** na barra lateral esquerda (ou pressione o atalho `Ctrl + Shift + X`).
   * Na barra de busca, digite **`Markdown PDF`** (desenvolvida por **yzane**).
   * Clique no botão azul **Install** (Instalar) e aguarde alguns segundos(a extensão baixará automaticamente uma instância leve do Chromium em segundo plano para renderizar os PDFs).
2. **Suporte a Fórmulas Matemáticas LaTeX no PDF (Dica Importante):**
   * Caso o seu relatório possua fórmulas matemáticas em LaTeX (`$$...$$`), você também pode instalar a extensão **`Markdown Preview Enhanced`** (por Yiyi Wang) ou adicionar o script do MathJax no template da extensão `Markdown PDF` para que todas as equações saiam perfeitamente formatadas no PDF.
3. **Exportar o seu Arquivo `.md` para `.pdf`:**
   * Abra o arquivo `relatorio_compreensao_pipelineXX.md` (ou `relatorio_final_pipelineXX.md`) no editor do VS Code.
   * Verifique se os caminhos das imagens estão corretos (ex.: `./images/figura01_pipeline03.jpg`).
   * Clique com o **botão direito do mouse** em qualquer lugar dentro do texto do arquivo `.md` aberto.
   * No menu que se abrir, clique na opção **`Markdown PDF: Export (pdf)`**.
   * Aguarde a notificação no canto inferior direito do VS Code informando que o PDF foi gerado. O arquivo `.pdf` aparecerá automaticamente na mesma pasta (`docs/pipeline_XX/`) com o mesmo nome do arquivo `.md`!

---

## 6. Como Enviar seu Trabalho (`git push`) e Abrir o Pull Request para a `develop`

Depois que você conferiu que todos os seus arquivos estão nas pastas certas e que **não alterou o `.gitignore`**, siga os passos abaixo para subir a sua entrega para o GitHub:

### Passo 6.1: Verificar os Arquivos Modificados
No terminal do VS Code, rode:
```bash
git status
```
Confira a lista de arquivos em vermelho. Veja se os seus scripts em `python_scripts/pipeline_XX/`, o seu CSV em `benchmarks/benchmark_pipeline_XX.csv` e os seus relatórios/imagens em `docs/pipeline_XX/` estão listados, e **certifique-se de que o `.gitignore` e a pasta `configs_scripts` NÃO estão sendo modificados/enviados**.

### Passo 6.2: Adicionar, Fazer o Commit e Enviar (`git push`)
1. Adicione todos os novos arquivos ao *stage* do Git:
   ```bash
   git add .
   ```
2. Crie um commit com uma mensagem clara identificando o seu pipeline:
   ```bash
   git commit -m "feat(pipeline-XX): adiciona scripts, benchmark csv e relatorios completos do pipeline XX"
   ```
3. Envie a sua branch para o repositório remoto no GitHub:
   ```bash
   git push -u origin feature-develop-seu_nickname_do_github
   ```

### Passo 6.3: Criar o Pull Request (PR) para a Branch `develop`
1. Abra o navegador e acesse a página do nosso repositório no **GitHub**.
2. Você verá um aviso amarelo no topo dizendo *"feature-develop-seu_nickname had recent pushes"* com um botão verde **`Compare & pull request`**. Clique nele (ou vá na aba **Pull requests** -> **New pull request**).
3. **ATENÇÃO MÁXIMA NAS BRANCHES DO PULL REQUEST:**
   * **base repository / base branch (Esquerda):** Selecione **`develop`** *(NUNCA deixe `main` como destino!)*.
   * **head repository / compare branch (Direita):** Selecione a sua branch **`feature-develop-seu_nickname_do_github`**.
   ```
   [ base: develop ]  <---  [ compare: feature-develop-seu_nickname_do_github ]
   ```
4. **Título do Pull Request:**
   * Coloque no padrão: `[Pipeline XX] Entrega Completa - <Seu Nome>`
5. **Descrição do Pull Request (Checklist de Entrega):**
   Copie e cole o checklist abaixo na descrição do PR marcando o que você concluiu:
   ```markdown
   ## Resumo da Entrega - Pipeline XX
   - **Responsável:** <Seu Nome>
   - **Estágio 1:** <Modelo 1>
   - **Estágio 2:** <Modelo 2> (Acurácia obtida: XX,XX%)
   - **Estágio 3:** <Modelo 3> (Latência média total: XXXX,XX ms)

   ## Checklist de Padronização
   - [x] Relatório de Compreensão (`.md` e `.pdf`) salvo em `docs/pipeline_XX/`
   - [x] Scripts de treino, cache `.npz` e pipeline unificado salvos em `python_scripts/pipeline_XX/`
   - [x] Arquivo de telemetria e resultados salvo em `benchmarks/benchmark_pipeline_XX.csv`
   - [x] Artigo Final Experimental (`.md` e `.pdf`) e figuras salvos em `docs/pipeline_XX/`
   - [x] Arquivo `.gitignore` mantido intacto e credenciais protegidas
   ```
6. Clique no botão verde **`Create pull request`** e avise no grupo do WhatsApp para revisarmos e fazermos o *merge* na `develop`!