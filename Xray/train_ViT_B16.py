"""
X-ray 기반 질병 다중분류 (5-class) 학습 코드
모델: Vision Transformer ViT-B/16 (ImageNet pretrained)
클래스: NORMAL, TUBERCULOSIS, COVID19, PNEUMONIA_virus, PNEUMONIA_bacteria

데이터 폴더 구조 (ImageFolder 형식):
    dataset/
    ├── NORMAL/
    ├── TUBERCULOSIS/
    ├── COVID19/
    ├── PNEUMONIA_virus/
    └── PNEUMONIA_bacteria/
"""

# ============================================================
# 0. CONFIG — 하이퍼파라미터 및 경로를 여기서 한 번에 수정
# ============================================================
CONFIG = {
    # 데이터 경로
    "data_dir": "dataset",

    # 클래스 정보
    "num_classes": 5,
    "class_names": [
        "NORMAL",
        "TUBERCULOSIS",
        "COVID19",
        "PNEUMONIA_virus",
        "PNEUMONIA_bacteria",
    ],

    # 이미지 크기 (ViT-B/16은 224x224 입력 기준으로 설계됨)
    "img_size": 224,

    # 학습 하이퍼파라미터
    # ViT는 파라미터가 많으므로 batch_size=16으로 GPU 메모리 부담 완화
    "batch_size": 16,
    # ViT fine-tuning에는 작은 lr이 안정적 (1e-4 ~ 5e-5 권장)
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,   # AdamW L2 정규화 → ViT 과적합 억제에 중요
    "num_epochs": 50,

    # 데이터 분할 비율 (합이 1.0)
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,

    # Early Stopping: val_loss가 이 횟수 동안 개선되지 않으면 학습 중단
    "early_stopping_patience": 10,

    # 재현성을 위한 랜덤 시드
    "seed": 42,

    # 모델 및 결과 저장 경로
    "save_path": "best_model_vit_b16.pth",
    "plot_path": "training_history_vit_b16.png",

    # DataLoader workers (Windows에서는 0 권장)
    "num_workers": 0,
}

# ============================================================
# 1. 라이브러리 Import
# ============================================================
import os
import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 2. 랜덤 시드 고정 — 실험 재현성 확보
# ============================================================
def set_seed(seed: int) -> None:
    """모든 난수 생성기의 시드를 고정하여 실험을 재현 가능하게 만듭니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN 비결정적 알고리즘 비활성화
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. GPU 확인
# ============================================================
def get_device() -> torch.device:
    """사용 가능한 GPU가 있으면 CUDA, 없으면 CPU를 반환합니다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 사용 디바이스: {device}")
    if device.type == "cuda":
        print(f"         GPU 이름: {torch.cuda.get_device_name(0)}")
        print(
            f"         GPU 메모리: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    return device


# ============================================================
# 4. Transform 정의
# ============================================================
def get_transforms(img_size: int) -> dict:
    """
    X-ray 이미지에 적합한 데이터 변환을 반환합니다.

    [클래스당 700장에서 Augmentation이 필요한 이유]
    총 ~3,500장(train ~2,450장)은 딥러닝 기준 소규모입니다.
    특히 ViT-B/16은 86M 파라미터로 CNN보다 훨씬 크기 때문에
    데이터 없이 학습하면 훈련 셋을 암기(memorize)하는 과적합 위험이 높습니다.
    약한 augmentation으로 가상의 다양성을 부여해 일반화 성능을 높여야 합니다.

    [과도한 Augmentation을 피해야 하는 이유]
    X-ray 진단에서 폐 음영 밀도, 병변의 위치·형태·크기가 핵심 단서입니다.
    - 수직 뒤집기: 심장·횡격막의 해부학적 위치가 역전 → 임상적으로 불가능
    - 강한 회전(>15도): 정면 촬영 표준에서 벗어나 병변 위치 관계 왜곡
    - 강한 ColorJitter: 방사선 투과도(폐 음영 밀도) 왜곡 → 병변 신호 손실
    - validation/test에는 augmentation을 절대 적용하지 않아야 실제 성능 측정 가능
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # X-ray는 좌우 대칭에 가까우므로 수평 뒤집기 허용
        transforms.RandomHorizontalFlip(p=0.5),
        # 회전을 ±10도로 제한: 흉부 X-ray 정면 촬영 기준 유지
        transforms.RandomRotation(degrees=10),
        # ColorJitter를 약하게: 폐 음영 과도 왜곡 방지
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Validation / Test: 크기 조정만 수행, augmentation 없음
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {
        "train": train_transform,
        "val":   val_test_transform,
        "test":  val_test_transform,
    }


# ============================================================
# 5. TransformDataset Wrapper — 데이터 누수 방지
# ============================================================
class TransformDataset(Dataset):
    """
    Subset에 독립적인 transform을 적용하는 Wrapper 클래스.

    [데이터 누수 방지]
    ImageFolder 하나를 split하면 모든 Subset이 같은 transform을 공유합니다.
    이 wrapper를 사용하면 train/val/test 각 Subset에 서로 다른 transform을
    독립적으로 적용하고, 동일 이미지가 여러 split에 중복 포함되는 일을 막습니다.
    """
    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        image, label = self.subset.dataset[self.subset.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 6. 데이터 분할 — Stratified Split으로 클래스 비율 유지
# ============================================================
def build_dataloaders(cfg: dict, transforms_dict: dict) -> tuple:
    """
    ImageFolder 데이터셋을 train/val/test로 분할하고 DataLoader를 반환합니다.

    [Stratified Split 전략]
    1. transform 없이 Raw ImageFolder를 먼저 로드합니다.
    2. sklearn의 train_test_split으로 클래스 비율(stratify)을 유지하며 분할합니다.
       → 각 클래스가 train/val/test에 70:15:15 비율로 균등하게 포함됩니다.
    3. 각 Subset에 TransformDataset을 씌워 transform을 독립 적용합니다.
    """
    raw_dataset = ImageFolder(root=cfg["data_dir"])

    print(f"\n[Dataset] 전체 이미지 수: {len(raw_dataset)}")
    print(f"[Dataset] 클래스 목록: {raw_dataset.classes}")
    for cls, idx in raw_dataset.class_to_idx.items():
        count = sum(1 for _, label in raw_dataset.samples if label == idx)
        print(f"          {cls}: {count}장")

    all_indices = list(range(len(raw_dataset)))
    all_labels  = [label for _, label in raw_dataset.samples]

    # 1단계: 전체 → train / (val+test) 분리
    val_test_ratio = cfg["val_ratio"] + cfg["test_ratio"]
    train_idx, valtest_idx = train_test_split(
        all_indices,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=cfg["seed"],
    )

    # 2단계: (val+test) → val / test 분리
    valtest_labels = [all_labels[i] for i in valtest_idx]
    val_ratio_within = cfg["val_ratio"] / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=(1.0 - val_ratio_within),
        stratify=valtest_labels,
        random_state=cfg["seed"],
    )

    print(
        f"\n[Split] Train: {len(train_idx)}장 | "
        f"Val: {len(val_idx)}장 | "
        f"Test: {len(test_idx)}장"
    )

    train_ds = TransformDataset(Subset(raw_dataset, train_idx), transforms_dict["train"])
    val_ds   = TransformDataset(Subset(raw_dataset, val_idx),   transforms_dict["val"])
    test_ds  = TransformDataset(Subset(raw_dataset, test_idx),  transforms_dict["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# 7. 모델 빌드 — ViT-B/16 Fine-tuning
# ============================================================
def build_model(num_classes: int) -> nn.Module:
    """
    ViT-B/16 (ImageNet pretrained)의 classification head를
    Linear(768 → num_classes)로 교체한 모델을 반환합니다.

    [ViT-B/16 구조 설명]
    - 입력 224x224를 16x16 패치 196개로 분할
    - 각 패치를 768차원 임베딩으로 변환 후 [CLS] 토큰 추가
    - 12개 Transformer Encoder 블록(Multi-Head Self-Attention + MLP)
    - [CLS] 토큰의 최종 표현을 heads.head(Linear)로 분류

    torchvision ViT-B/16 classifier 구조:
        model.heads = Sequential(
            OrderedDict([("head", Linear(768, 1000))])
        )
    → model.heads.head만 교체하여 원래 attention 블록은 모두 유지합니다.

    [Dropout]
    ViT-B/16 내부 Transformer 블록에는 attention_dropout과 mlp_dropout이
    이미 포함되어 있습니다. 소규모 데이터 과적합 억제를 위해 weight_decay와
    Early Stopping을 함께 사용합니다.
    """
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    # heads.head의 in_features 확인 (ViT-B/16: 768)
    in_features = model.heads.head.in_features

    # classification head 교체 (기존 heads Sequential 구조 유지)
    model.heads.head = nn.Linear(in_features, num_classes)

    print(f"\n[Model] ViT-B/16 로드 완료 (pretrained=ImageNet)")
    print(f"[Model] heads.head: Linear({in_features}, {num_classes})")
    return model


# ============================================================
# 8. 학습 함수 (1 epoch)
# ============================================================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    모델을 1 epoch 동안 학습하고 (평균 loss, accuracy)를 반환합니다.

    Returns:
        avg_loss (float): 배치 평균 Cross-Entropy Loss
        accuracy (float): 정확도 (0 ~ 1)
    """
    model.train()  # Dropout, LayerNorm을 학습 모드로 전환

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)             # shape: (batch, num_classes)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ============================================================
# 9. 검증 / 테스트 공용 평가 함수
# ============================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list, list]:
    """
    모델을 평가 모드에서 실행하고 (loss, accuracy, 예측값, 정답)을 반환합니다.

    @torch.no_grad()로 gradient 계산 비활성화 → 메모리 절약 및 속도 향상.

    Returns:
        avg_loss (float): 배치 평균 loss
        accuracy (float): 정확도 (0 ~ 1)
        all_preds (list): 예측 결과
        all_labels (list): 실제 라벨
    """
    model.eval()  # Dropout 비활성화, LayerNorm 추론 모드

    total_loss    = 0.0
    correct       = 0
    total_samples = 0
    all_preds     = []
    all_labels    = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += images.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy, all_preds, all_labels


# ============================================================
# 10. Test 최종 평가 — 지표 출력 및 Confusion Matrix 시각화
# ============================================================
def evaluate_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list,
    save_dir: str = ".",
) -> None:
  
    _, accuracy, all_preds, all_labels = evaluate(model, loader, criterion, device)

    print("\n" + "=" * 60)
    print("[ Test Set 최종 평가 결과 — ViT-B/16 ]")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    ))

    # Confusion Matrix 시각화
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Confusion Matrix (Test Set) — ViT-B/16", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix_vit_b16.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n[저장] Confusion Matrix → {cm_path}")


# ============================================================
# 11. 학습 결과 그래프 저장
# ============================================================
def plot_training_history(history: dict, save_path: str) -> None:
    """
    epoch별 train/val loss와 accuracy 곡선을 한 그림에 저장합니다.

    Args:
        history: {"train_loss": [...], "val_loss": [...],
                  "train_acc": [...], "val_acc": [...]}
        save_path: 저장 경로 (예: "training_history_vit_b16.png")
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — ViT-B/16", fontsize=15)

    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Acc", marker="o", markersize=3)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val Acc",   marker="o", markersize=3)
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[저장] Training History 그래프 → {save_path}")


# ============================================================
# 12. 메인 학습 루프
# ============================================================
def main() -> None:
    cfg = CONFIG

    # ── 시드 및 디바이스 초기화 ──────────────────────────────
    set_seed(cfg["seed"])
    device = get_device()

    # ── Transform 및 DataLoader 준비 ─────────────────────────
    transforms_dict = get_transforms(cfg["img_size"])
    train_loader, val_loader, test_loader = build_dataloaders(cfg, transforms_dict)

    # ── 모델 생성 및 GPU 이동 ─────────────────────────────────
    model = build_model(cfg["num_classes"])
    model = model.to(device)

    # ── 손실 함수 ─────────────────────────────────────────────
    # 단일 라벨 다중분류 → CrossEntropyLoss (Softmax 내장)
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer ────────────────────────────────────────────
    # AdamW: weight_decay로 L2 정규화 적용
    # ViT는 파라미터가 많아(86M) weight_decay가 과적합 억제에 특히 중요
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Scheduler: CosineAnnealingLR ─────────────────────────
    # learning rate를 코사인 곡선으로 서서히 감소시켜
    # 학습 후반부 세밀한 수렴을 유도합니다.
    # ViT fine-tuning에서 lr이 너무 크면 pretrained feature가 손상됩니다.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["num_epochs"],
        eta_min=1e-6,
    )

    # ── 학습 기록 초기화 ──────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss  = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_count = 0

    print("\n" + "=" * 60)
    print("[ 학습 시작 — ViT-B/16 ]")
    print("=" * 60)

    # ── Epoch 루프 ───────────────────────────────────────────
    for epoch in range(1, cfg["num_epochs"] + 1):
        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        # Scheduler 스텝
        scheduler.step()

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:>3}/{cfg['num_epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        # Best 모델 저장 — val_loss 개선 시에만 저장
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, cfg["save_path"])
            print(f"  → Best model 저장 (val_loss: {best_val_loss:.4f})")
            patience_count = 0
        else:
            patience_count += 1

        # Early Stopping
        if patience_count >= cfg["early_stopping_patience"]:
            print(
                f"\n[Early Stopping] {cfg['early_stopping_patience']} epochs 동안 "
                "val_loss 개선 없음. 학습 조기 종료."
            )
            break

    print("\n[학습 완료]")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # ── 그래프 저장 ──────────────────────────────────────────
    plot_training_history(history, cfg["plot_path"])

    # ── 최종 평가 (best model 가중치 복원 후 test) ─────────────
    print("\n[Test 평가] best_model_vit_b16.pth 가중치로 복원 중...")
    model.load_state_dict(best_model_wts)

    evaluate_test(
        model,
        test_loader,
        criterion,
        device,
        cfg["class_names"],
        save_dir=".",
    )


# ============================================================
# 13. 진입점
# ============================================================
if __name__ == "__main__":
    main()
