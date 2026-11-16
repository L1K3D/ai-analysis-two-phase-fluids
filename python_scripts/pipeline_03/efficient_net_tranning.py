import os
import sys
import numpy as np
from pathlib import Path

# Localiza a raiz do projeto automaticamente para importar a pasta utils
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

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Rescaling, BatchNormalization
from tensorflow.keras.optimizers import Adam

# ==========================================
# 1. INGESTÃO E DIVISÃO (TREINO ENXUTO vs TESTE MASSIVO)
# ==========================================
# DICA: Se quiser puxar TODAS as imagens da pasta do Drive sem limite, 
# basta trocar limit_per_class_collected=50 por limit_per_class_collected=None
LIMITE_POR_CLASSE = 50 

print(f"📥 [1/4] Baixando imagens do Google Drive (Limite por classe: {LIMITE_POR_CLASSE})...")
X_exp, y_exp = build_dataset(limit_per_class_collected=LIMITE_POR_CLASSE, folder_name_collected="expulsion")
X_inc, y_inc = build_dataset(limit_per_class_collected=LIMITE_POR_CLASSE, folder_name_collected="incubation")

X_completo = np.concatenate((X_exp, X_inc), axis=0)
y_completo = np.concatenate((y_exp, y_inc), axis=0)

# Treina com lote menor (30%) e separa a maioria (70%) para o teste do Pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X_completo, y_completo, test_size=0.70, stratify=y_completo, random_state=42
)

# Salva o lote de teste no disco para o pipeline_03.py não precisar baixar do Drive de novo
caminho_cache_teste = SCRIPT_DIR / "dados_teste_p3.npz"
np.savez_compressed(caminho_cache_teste, X_test=X_test, y_test=y_test)
print(f"📦 Lote dividido: {len(X_train)} imagens para Treino | {len(X_test)} imagens salvas para Teste.")

# ==========================================
# 2. ARQUITETURA COM CORREÇÃO DE ESCALA [0, 255]
# ==========================================
print("🧠 [2/4] Montando a EfficientNet-B0 corrigida para a escala [0, 255]...")
efficient_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Descongela as últimas 20 camadas, mas mantém BatchNormalization congelada para não viciar numa classe só
efficient_base.trainable = True
for layer in efficient_base.layers[:-20]:
    layer.trainable = False
for layer in efficient_base.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = False

modelo = Sequential([
    # Converte automaticamente de [0.0, 1.0] (do build_dataset) para [0.0, 255.0]
    Rescaling(255.0, input_shape=(224, 224, 3)),
    efficient_base,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid') # 0 = Expulsão, 1 = Incubação
])

modelo.compile(
    optimizer=Adam(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 3. TREINAMENTO E SALVAMENTO
# ==========================================
print("🔥 [3/4] Treinando no lote enxuto (12 épocas)...")
modelo.fit(
    X_train, y_train,
    epochs=12,
    batch_size=8,
    validation_data=(X_test[:20], y_test[:20]), # Valida rápido nas primeiras 20 de teste
    verbose=1
)

caminho_modelo = SCRIPT_DIR / "efficientnet_finetuned.keras"
print("💾 [4/4] Salvando modelo e cache...")
modelo.save(caminho_modelo)
print(f"✅ Pronto! Modelo salvo em: {caminho_modelo}")