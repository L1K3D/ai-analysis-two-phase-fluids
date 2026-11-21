import sys
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

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

install_missing_libs(["torch", "torchvision", "scikit-learn", "numpy"])

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models


def set_batchnorm_eval(module):
    if isinstance(module, nn.BatchNorm2d):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False


def build_resnet18_classifier():
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    for name, param in model.named_parameters():
        if not name.startswith("layer4"):
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return model


def preprocess_batch_imagenet(x_nhwc: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(x_nhwc).permute(0, 3, 1, 2).float().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Dispositivo de treinamento: {device}")

    LIMITE_POR_CLASSE = 50
    print(f"[1/4] Baixando imagens do Google Drive (Limite por classe: {LIMITE_POR_CLASSE})...")
    X_exp, y_exp = build_dataset(limit_per_class_collected=LIMITE_POR_CLASSE, folder_name_collected="expulsion")
    X_inc, y_inc = build_dataset(limit_per_class_collected=LIMITE_POR_CLASSE, folder_name_collected="incubation")

    X_completo = np.concatenate((X_exp, X_inc), axis=0)
    y_completo = np.concatenate((y_exp, y_inc), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X_completo, y_completo, test_size=0.70, stratify=y_completo, random_state=42
    )

    caminho_cache_teste = SCRIPT_DIR / "dados_teste_p2.npz"
    np.savez_compressed(caminho_cache_teste, X_test=X_test, y_test=y_test)
    print(f"Lote dividido: {len(X_train)} imagens Treino | {len(X_test)} imagens Teste salvas em {caminho_cache_teste.name}")

    print("[2/4] Montando a arquitetura ResNet-18 (Transfer Learning)...")
    model = build_resnet18_classifier().to(device)

    X_train_t = preprocess_batch_imagenet(X_train, device)
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1).to(device)

    X_val_t = preprocess_batch_imagenet(X_test[:20], device)
    y_val_t = torch.from_numpy(y_test[:20]).float().unsqueeze(1).to(device)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=8, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)

    print("[3/4] Treinando no lote enxuto (12 épocas)...")
    for epoch in range(1, 13):
        t0 = time.perf_counter()
        model.train()
        model.apply(set_batchnorm_eval)

        train_loss, correct_train, total_train = 0.0, 0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * bx.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)

        train_loss /= total_train
        acc_train = (correct_train / total_train) * 100.0

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = (torch.sigmoid(val_logits) > 0.5).float()
            acc_val = ((val_preds == y_val_t).sum().item() / len(y_val_t)) * 100.0

        dt = time.perf_counter() - t0
        print(
            f"Época {epoch:02d}/12 | Tempo: {dt:.2f}s | "
            f"Loss Treino: {train_loss:.4f} | Acc Treino: {acc_train:.2f}% | "
            f"Loss Val: {val_loss:.4f} | Acc Val: {acc_val:.2f}%"
        )

    caminho_modelo = SCRIPT_DIR / "resnet18_finetuned.pth"
    torch.save(model.state_dict(), caminho_modelo)
    print(f"[4/4] Pesos salvos com sucesso em: {caminho_modelo}")
