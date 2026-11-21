import sys
import time
import argparse
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

install_missing_libs(["torch", "torchvision", "ultralytics", "requests", "pandas", "opencv-python", "psutil"])

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO


class UnifiedPipeline07:
    """Pipeline 07: YOLOv8n-seg + ResNet-18 Fine-Tuned + LLaMA-3-8B (Puramente Textual via Ollama)."""

    def __init__(
        self,
        yolo_weights: str = "yolov8n-seg.pt",
        resnet_weights_path: Path = project_root / "python_scripts" / "pipeline_02" / "resnet18_finetuned.pth",
        ollama_model: str = "llama3",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.segmenter = YOLO(yolo_weights)
        self.classifier = self._load_resnet18(Path(resnet_weights_path))

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
        if not weights_path.is_file():
            raise FileNotFoundError(f"Pesos da ResNet-18 não encontrados em: {weights_path}")
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def warmup(self):
        print(f"Aquecendo motores (Warm-up YOLOv8n-seg + ResNet-18 + {self.ollama_model})...")
        dummy_img = np.zeros((224, 224, 3), dtype=np.float32)
        _ = self._stage1_yolov8_seg(dummy_img)
        _ = self._stage2_resnet18(dummy_img)

        try:
            self.http_session.post(
                self.ollama_url,
                json={
                    "model": self.ollama_model,
                    "prompt": "Reply with one word: READY.",
                    "stream": False,
                    "keep_alive": "20m",
                    "options": {"num_predict": 5, "temperature": 0.0},
                },
                timeout=180,
            )
        except Exception as e:
            print(f"Aviso no Warm-up do Ollama ({self.ollama_model}): {e}")

    def _stage1_yolov8_seg(self, img_norm: np.ndarray):
        t0 = time.perf_counter()
        img_bgr = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        results = self.segmenter.predict(img_bgr, imgsz=224, verbose=False)

        qtd_bolhas = 0
        pct_vapor = 0.0
        res = results[0]

        if res.masks is not None and len(res.masks.data) > 0:
            masks_np = res.masks.data.cpu().numpy()
            qtd_bolhas = int(masks_np.shape[0])
            union_mask = np.zeros((224, 224), dtype=np.uint8)
            for m in masks_np:
                m_resized = cv2.resize(m, (224, 224), interpolation=cv2.INTER_NEAREST)
                union_mask = np.maximum(union_mask, (m_resized > 0.5).astype(np.uint8))
            pct_vapor = round((float(np.sum(union_mask)) / (224 * 224)) * 100.0, 2)

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

    def _stage3_llama3_text(self, classe_ia: int, prob: float, qtd: int, pct: float):
        t0 = time.perf_counter()
        regime = "Incubation (stable bubbly nucleation)" if classe_ia == 1 else "Expulsion (vapor slug/churn instability)"
        conf_pct = round((prob if classe_ia == 1 else (1.0 - prob)) * 100, 1)

        prompt = (
            "You are a nuclear thermal-hydraulics safety assistant monitoring a Natural Circulation Loop (CCN). "
            f"Deep Vision Classifier (ResNet-18): {regime} with {conf_pct}% confidence. "
            f"Instance Segmentation Telemetry (YOLOv8n-seg): {qtd} bubbles detected, {pct}% planar void fraction. "
            "Based strictly on these numerical parameters, state the operational status "
            "(Nominal Operation, Operational Attention, or Critical Interlock) and give 1 concise technical justification sentence."
        )

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "20m",
            "options": {"num_predict": 60, "temperature": 0.1},
        }

        try:
            resp = self.http_session.post(self.ollama_url, json=payload, timeout=180)
            resp.raise_for_status()
            acao = resp.json().get("response", "").strip().replace("\n", " ")
        except requests.exceptions.RequestException as e:
            acao = f"ERRO Ollama: {e}"

        lat_ms = round((time.perf_counter() - t0) * 1000, 2)
        return acao, lat_ms

    def run_single_image(self, img_norm: np.ndarray):
        psutil.cpu_percent(interval=None)
        t_inicio = time.perf_counter()

        qtd, pct, lat_s1 = self._stage1_yolov8_seg(img_norm)
        classe, prob, lat_s2 = self._stage2_resnet18(img_norm)
        acao, lat_s3 = self._stage3_llama3_text(classe, prob, qtd, pct)

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
            "Hardware_Specs": self.hw_specs,
        }


if __name__ == "__main__":
    default_dataset = project_root / "python_scripts" / "pipeline_02" / "dados_teste_p2.npz"
    default_weights = project_root / "python_scripts" / "pipeline_02" / "resnet18_finetuned.pth"

    parser = argparse.ArgumentParser(description="Executa o Pipeline 07 (YOLOv8n-seg + ResNet-18 + LLaMA-3-8B)")
    parser.add_argument("--yolo", type=str, default="yolov8n-seg.pt", help="Pesos do YOLOv8-seg")
    parser.add_argument("--dataset", type=Path, default=default_dataset, help="Caminho para o arquivo .npz de teste")
    parser.add_argument("--weights", type=Path, default=default_weights, help="Caminho para os pesos .pth da ResNet-18")
    parser.add_argument("--model", type=str, default="llama3", help="Modelo textual do Ollama (ex: llama3, llama3:8b)")
    parser.add_argument("--output", type=Path, default=project_root / "benchmarks" / "benchmark_pipeline_07.csv", help="CSV de saída")
    args = parser.parse_args()

    dados = np.load(args.dataset)
    X_test, y_test = dados["X_test"], dados["y_test"]

    pipeline = UnifiedPipeline07(yolo_weights=args.yolo, resnet_weights_path=args.weights, ollama_model=args.model)
    pipeline.warmup()

    resultados = []
    for i in range(len(X_test)):
        metricas = pipeline.run_single_image(X_test[i])
        real_str = "Incubação" if y_test[i] == 1 else "Expulsão"
        resultados.append({"ID_Teste": i, "Real": real_str, **metricas})
        print(
            f"-> [{i+1:02d}/{len(X_test)}] Real: {real_str:9s} | "
            f"Pred: {metricas['Predicao_Classificador']:9s} | "
            f"Bolhas: {metricas['Qtd_Bolhas']} | Total: {metricas['Latencia_ms']} ms"
        )

    df = pd.DataFrame(resultados)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nBenchmark salvo com sucesso em: {args.output}")
