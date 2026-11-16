import os
import sys
from pathlib import Path

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

install_missing_libs(['requests', 'pandas', 'tensorflow', 'scikit-learn', 'opencv-python', 'psutil'])

import time
import platform
import psutil
import cv2
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

#---###---#

class UnifiedPipeline03:
    """
    Consolida os 3 estágios do Pipeline 03 (OpenCV + EfficientNet-B0 + LLaMA-3-8B)
    em um único objeto de inferência com telemetria de hardware embutida.
    """
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
        """Aquece a rede visual e mantém o LLaMA-3 carregado na RAM antes do cronômetro."""
        print("🔥 Aquecendo motores (Warm-up da EfficientNet e do Ollama)...")
        dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
        _ = self.classifier.predict(dummy_img, verbose=0)
        try:
            self.http_session.post(self.ollama_url, json={
                "model": self.ollama_model,
                "prompt": "Reply OK.",
                "stream": False,
                "keep_alive": "15m",
                "options": {"num_predict": 5}
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
        # A camada Rescaling(255.0) já está embutida dentro do modelo salvo!
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
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "15m",
            "options": {"num_predict": 45, "temperature": 0.1}
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
        psutil.cpu_percent(interval=None) # Zera contador de CPU para medir a janela da imagem
        t_inicio = time.perf_counter()

        qtd, pct, lat_cv = self._stage1_opencv(img_norm)
        classe, prob, lat_eff = self._stage2_efficientnet(img_norm)
        acao, lat_llm = self._stage3_llama3(classe, qtd, pct)

        lat_total = round((time.perf_counter() - t_inicio) * 1000, 2)
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = psutil.virtual_memory().percent

        return {
            "Predicao_EfficientNet": "Incubação" if classe == 1 else "Expulsão",
            "Probabilidade_Classe_1": prob,
            "Qtd_Bolhas": qtd,
            "Vapor_%": pct,
            "Acao_Sugerida": acao,
            "Lat_OpenCV_ms": lat_cv,
            "Lat_EfficientNet_ms": lat_eff,
            "Lat_LLaMA3_ms": lat_llm,
            "Latencia_ms": lat_total,
            "CPU_Uso_%": cpu_pct,
            "RAM_Uso_%": ram_pct,
            "Hardware_Specs": self.hw_specs
        }


# ==========================================
# EXECUÇÃO PRINCIPAL DO BENCHMARK
# ==========================================
if __name__ == "__main__":
    caminho_cache = SCRIPT_DIR / "dados_teste_p3.npz"
    caminho_modelo = SCRIPT_DIR / "efficientnet_finetuned.keras"

    if caminho_cache.exists():
        print("⚡ [1/4] Carregando lote de teste direto do cache local (Zero espera de rede)...")
        dados = np.load(caminho_cache)
        X_test, y_test = dados["X_test"], dados["y_test"]
    else:
        print("📥 [1/4] Cache não encontrado. Baixando imagens do Google Drive...")
        X_exp, y_exp = build_dataset(limit_per_class_collected=50, folder_name_collected="expulsion")
        X_inc, y_inc = build_dataset(limit_per_class_collected=50, folder_name_collected="incubation")
        X_completo = np.concatenate((X_exp, X_inc), axis=0)
        y_completo = np.concatenate((y_exp, y_inc), axis=0)
        _, X_test, _, y_test = train_test_split(
            X_completo, y_completo, test_size=0.70, stratify=y_completo, random_state=42
        )

    print(f"🧠 [2/4] Inicializando UnifiedPipeline03 ({len(X_test)} imagens de teste)...")
    pipeline = UnifiedPipeline03(keras_model_path=caminho_modelo)
    pipeline.warmup()

    resultados_p3 = []
    print("⏱️ [3/4] Iniciando cronometragem e telemetria do Pipeline 03...")

    for i in range(len(X_test)):
        metricas = pipeline.run_single_image(X_test[i])
        real_str = "Incubação" if y_test[i] == 1 else "Expulsão"

        linha = {
            "ID_Teste": i,
            "Real": real_str,
            **metricas
        }
        resultados_p3.append(linha)

        print(
            f"  -> [{i+1}/{len(X_test)}] Real: {real_str:9s} | "
            f"Pred: {metricas['Predicao_EfficientNet']:9s} (p={metricas['Probabilidade_Classe_1']:.2f}) | "
            f"Total: {metricas['Latencia_ms']} ms (LLM: {metricas['Lat_LLaMA3_ms']} ms) | "
            f"CPU: {metricas['CPU_Uso_%']}%"
        )

    df_p3 = pd.DataFrame(resultados_p3)
    caminho_csv = project_root / "benchmark_pipeline_03.csv"
    df_p3.to_csv(caminho_csv, index=False)

    acuracia = (df_p3["Real"] == df_p3["Predicao_EfficientNet"]).mean() * 100
    print(f"\n✅ [4/4] Benchmark concluído! Salvo em: {caminho_csv}")
    print(f"🎯 Acurácia da EfficientNet no lote de teste ({len(df_p3)} imgs): {acuracia:.2f}%")
    print(f"📊 Latência Média Total: {df_p3['Latencia_ms'].mean():.2f} ms")