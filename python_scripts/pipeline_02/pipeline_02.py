import sys
import time
import base64
import platform
import psutil
import cv2
import requests
import pandas as pd
import numpy as np
from pathlib import Path

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

install_missing_libs(["torch", "torchvision", "requests", "pandas", "opencv-python", "psutil"])

import torch
import torch.nn as nn
from torchvision import models


class UnifiedPipeline02:
    def __init__(self, resnet_weights_path: Path, ollama_model: str = "moondream"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self._load_resnet18(resnet_weights_path)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.ollama_url = "http://localhost:11434/api/generate"
        self.ollama_model = ollama_model
        self.http_session = requests.Session()

        self.hw_specs = (
            f"Device: {self.device.type.upper()} | "
            f"CPU: {platform.processor()} ({psutil.cpu_count(logical=False)}C/{psutil.cpu_count(logical=True)}T) | "
            f"RAM: {round(psutil.virtual_memory().total / (1024**3), 1)}GB"
        )

    def _load_resnet18(self, weights_path: Path):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def warmup(self):
        print("Aquecendo motores (Warm-up ResNet-18 + Moondream2)...")
        dummy_img = np.zeros((224, 224, 3), dtype=np.float32)
        _ = self._stage2_resnet18(dummy_img)

        dummy_uint8 = np.zeros((224, 224, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy_uint8)
        dummy_b64 = base64.b64encode(buf).decode("utf-8")

        try:
            self.http_session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": "Reply OK.",
                    "images": [dummy_b64],
                    "stream": False,
                    "keep_alive": "15m",
                    "options": {"num_predict": 5}
                },
                timeout=90
            )
        except Exception as e:
            print(f"Aviso no Warm-up do Ollama: {e}")

    def _stage1_opencv(self, img_norm: np.ndarray):
        t0 = time.perf_counter()
        img_uint8 = (img_norm * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        bordas = cv2.Canny(gray, 50, 150)
        contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        qtd_bolhas = len(contornos)
        pct_vapor = round((sum(cv2.contourArea(c) for c in contornos) / (224 * 224)) * 100, 2)
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return qtd_bolhas, pct_vapor, lat_ms

    def _stage2_resnet18(self, img_norm: np.ndarray):
        t0 = time.perf_counter()
        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            logit = self.classifier(tensor)
            prob = float(torch.sigmoid(logit).item())
        classe = 1 if prob > 0.5 else 0
        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return classe, round(prob, 4), lat_ms

    def _stage3_moondream(self, img_norm: np.ndarray, classe_ia: int, prob: float, qtd: int, pct: float):
        t0 = time.perf_counter()
        estado = "Incubação (bubbly nucleation)" if classe_ia == 1 else "Expulsão (vapor slug/churn)"
        conf_pct = round((prob if classe_ia == 1 else (1.0 - prob)) * 100, 1)

        img_bgr = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            f"You are a nuclear thermal-hydraulics safety assistant monitoring a Natural Circulation Loop. "
            f"Classifier output: {estado} (confidence: {conf_pct}%). "
            f"OpenCV geometric telemetry: {qtd} contours detected, {pct}% planar void fraction. "
            f"Inspect the image directly and verify consistency with the classifier. "
            f"Provide 1 concise operational command sentence (Nominal Operation, Operational Attention, or Critical Interlock)."
        )

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "keep_alive": "15m",
            "options": {"num_predict": 45, "temperature": 0.1}
        }

        try:
            resp = self.http_session.post(self.ollama_url, json=payload, timeout=120)
            resp.raise_for_status()
            acao = resp.json().get("response", "").strip().replace("\n", " ")
        except requests.exceptions.RequestException as e:
            acao = f"ERRO Ollama: {e}"

        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return acao, lat_ms

    def run_single_image(self, img_norm: np.ndarray):
        psutil.cpu_percent(interval=None)
        t_inicio = time.perf_counter()

        qtd, pct, lat_s1 = self._stage1_opencv(img_norm)
        classe, prob, lat_s2 = self._stage2_resnet18(img_norm)
        acao, lat_s3 = self._stage3_moondream(img_norm, classe, prob, qtd, pct)

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
    caminho_cache = SCRIPT_DIR / "dados_teste_p2.npz"
    caminho_modelo = SCRIPT_DIR / "resnet18_finetuned.pth"

    dados = np.load(caminho_cache)
    X_test, y_test = dados["X_test"], dados["y_test"]

    pipeline = UnifiedPipeline02(resnet_weights_path=caminho_modelo, ollama_model="moondream")
    pipeline.warmup()

    resultados = []
    for i in range(len(X_test)):
        metricas = pipeline.run_single_image(X_test[i])
        real_str = "Incubação" if y_test[i] == 1 else "Expulsão"
        resultados.append({"ID_Teste": i, "Real": real_str, **metricas})
        print(
            f"-> [{i+1:02d}/{len(X_test)}] Real: {real_str:9s} | "
            f"Pred: {metricas['Predicao_Classificador']:9s} | Total: {metricas['Latencia_ms']} ms"
        )

    df = pd.DataFrame(resultados)
    pasta_benchmarks = project_root / "benchmarks"
    pasta_benchmarks.mkdir(exist_ok=True)

    caminho_csv = pasta_benchmarks / "benchmark_pipeline_02.csv"
    df.to_csv(caminho_csv, index=False)
    print(f"\nBenchmark salvo com sucesso em: {caminho_csv}")
