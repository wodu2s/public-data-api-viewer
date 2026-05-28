"""
X-ray 기반 질병 다중분류 (5-class) 학습 코드
모델: EfficientNet-B3 (ImageNet pretrained)
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

    # 이미지 크기
    "img_size": 224,

    # 학습 하이퍼파라미터
    # 클래스당 ~700장, 총 ~3,500장 규모이므로 batch_size=16으로 설정
    # GPU 메모리 여유가 있다면 32로 올려도 무방
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,   # AdamW L2 정규화 → 과적합 억제
    "num_epochs": 50,

    # 데이터 분할 비율 (합이 1.0)
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,

    # Early Stopping: val_loss가 이 횟수 동안 개선되지 않으면 학습 중단
    # 클래스당 700장의 소규모 데이터에서는 과도한 학습 방지가 중요
    "early_stopping_patience": 10,

    # 재현성을 위한 랜덤 시드
    "seed": 42,

    # 모델 및 결과 저장 경로
    "save_path": "best_model_effb3.pth",
    "plot_path": "training_history_effb3.png",

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
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

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
    # cuDNN 비결정적 알고리즘 비활성화 (속도보다 재현성 우선)
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
    클래스당 700장(총 3,500장)은 딥러닝 기준으로 소규모입니다.
    충분한 데이터 없이 학습하면 모델이 훈련 셋을 통째로 암기(memorize)하여
    훈련 정확도는 높지만 검증/테스트 성능이 낮은 과적합이 발생합니다.
    수평 뒤집기, 소폭 회전 등 약한 augmentation은 동일 이미지에서 다양한
    변형을 생성해 모델이 더 일반화된 특징을 학습하도록 도와줍니다.

    [과도한 Augmentation을 피해야 하는 이유]
    X-ray 진단에서 폐 음영 밀도, 병변의 위치·형태가 핵심 단서입니다.
    - 수직 뒤집기: 심장·횡격막의 해부학적 위치가 역전되어 임상적으로 불가능
    - 강한 회전(>15도): 표준 정면 촬영 자세를 벗어나 병변 특징 왜곡
    - 강한 ColorJitter: 폐 음영 밀도(방사선 투과도)를 왜곡해 병변 신호 손실
    따라서 validation/test에는 augmentation을 전혀 적용하지 않습니다.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet 통계값
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # 수평 뒤집기: X-ray는 좌우 대칭에 가까우므로 허용
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
# 5. TransformDataset Wrapper — 데이터 누수 방지의 핵심
# ============================================================
class TransformDataset(Dataset):
    """
    Subset에 독립적인 transform을 적용하는 Wrapper 클래스.

    [데이터 누수 방지]
    ImageFolder 하나를 split하면 모든 Subset이 같은 transform을 공유합니다.
    이 wrapper를 사용하면 train Subset에는 augmentation transform을,
    val/test Subset에는 기본 transform을 각각 독립적으로 적용할 수 있습니다.
    동일 이미지가 train·val·test에 중복 포함되는 일도 없습니다.
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
    1. transform 없이 Raw ImageFolder를 로드합니다.
    2. sklearn의 train_test_split으로 클래스 비율(stratify)을 유지하며 분할합니다.
       → 클래스당 약 490장(train), 105장(val), 105장(test)으로 균등 분할됩니다.
    3. 각 Subset에 TransformDataset을 씌워 transform을 독립 적용합니다.
    """
    # transform 없이 먼저 로드 (인덱스 기반 split을 위해)
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

    # Subset + TransformDataset 생성
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
# 7. 모델 빌드 — EfficientNet-B3 Fine-tuning
# ============================================================
def build_model(num_classes: int) -> nn.Module:
    """
    EfficientNet-B3 (ImageNet pretrained)의 classifier를
    Dropout(0.3) + Linear(1536 → num_classes)로 교체한 모델을 반환합니다.

    [EfficientNet-B3를 선택한 이유]
    - Compound Scaling: 네트워크의 depth(깊이)·width(채널 수)·resolution(해상도)을
      균형 있게 동시에 확장하는 방식으로, 같은 파라미터 대비 최고 성능을 냅니다.
    - B3 규모의 적절성: 약 12M 파라미터로 B0(5M)보다 표현력이 높으면서
      B7(66M)보다 훨씬 가볍습니다. 클래스당 700장의 소규모 데이터에서
      pretrained feature를 최대한 활용할 수 있는 균형잡힌 선택입니다.
    - 높은 파라미터 효율: EfficientNet은 동일 FLOPs 기준 ImageNet 정확도가
      ResNet, DenseNet보다 우수하여 의료영상 전이학습에 폭넓게 활용됩니다.

    [Dropout 명시적 추가]
    EfficientNet-B3의 원래 classifier는 Sequential(Dropout(0.3), Linear(1536, 1000))
    입니다. 클래스 수를 교체하면서 Dropout도 함께 재정의하여 정규화 효과를 유지합니다.
    """
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

    # 원래 classifier의 in_features 확인 (EfficientNet-B3: 1536)
    in_features = model.classifier[1].in_features

    # Dropout(0.3) + Linear 으로 교체
    # Dropout: 학습 중 뉴런을 무작위 비활성화 → 과적합 방지 (소규모 데이터에 특히 중요)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    print(f"\n[Model] EfficientNet-B3 로드 완료 (pretrained=ImageNet)")
    print(f"[Model] Classifier: Dropout(0.3) + Linear({in_features}, {num_classes})")
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
    model.train()  # Dropout, BatchNorm을 학습 모드로 전환

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)             # shape: (batch, num_classes)
        loss = criterion(outputs, labels)   # CrossEntropyLoss

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
    validation과 test에서 공통으로 사용합니다.

    Returns:
        avg_loss (float): 배치 평균 loss
        accuracy (float): 정확도 (0 ~ 1)
        all_preds (list): 배치 예측 결과
        all_labels (list): 실제 라벨
    """
    model.eval()  # Dropout 비활성화, BatchNorm 추론 모드 전환

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
    """
    테스트 셋에 대한 최종 평가를 수행하고 결과를 출력합니다.

    출력 지표:
    - Overall Accuracy
    - Per-class Precision, Recall, F1-score (classification_report)
    - Confusion Matrix (seaborn heatmap으로 저장)

    [Accuracy만 보면 안 되는 이유]
    의료 진단 맥락에서 오분류의 비용은 클래스마다 다릅니다.
    - False Negative (환자를 정상으로 분류): 치료 기회 상실 → 비용 매우 높음
    - False Positive (정상을 환자로 분류): 추가 검사로 이어질 뿐 → 상대적으로 낮음
    따라서 Recall(민감도 = TP/(TP+FN))이 특히 중요하며,
    Precision·F1·Confusion Matrix를 함께 분석해야 모델의 임상 유용성을 판단할 수 있습니다.
    예: PNEUMONIA_virus와 PNEUMONIA_bacteria 간 혼동이 많은지 파악 가능.

    [CrossEntropyLoss를 사용하는 이유]
    이 태스크는 "단일 라벨 다중분류"입니다. 각 X-ray는 하나의 진단만 가집니다.
    CrossEntropyLoss는 내부적으로 Softmax를 적용해 5개 클래스의 확률 합=1 제약을 부여하고,
    정답 클래스의 로그 확률을 최대화합니다.
    BCEWithLogitsLoss는 각 클래스를 독립적인 이진 분류로 취급(Sigmoid 내장)하므로
    한 샘플에 여러 정답이 가능한 "다중 라벨(multi-label)" 분류에만 적합합니다.
    """
    _, accuracy, all_preds, all_labels = evaluate(model, loader, criterion, device)

    print("\n" + "=" * 60)
    print("[ Test Set 최종 평가 결과 ]")
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
    ax.set_title("Confusion Matrix (Test Set) — EfficientNet-B3", fontsize=13, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix_effb3.png")
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
        save_path: 저장 경로 (예: "training_history_effb3.png")
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — EfficientNet-B3", fontsize=15)

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
    # AdamW: weight_decay로 L2 정규화를 적용해 과적합을 억제
    # 클래스당 700장 소규모 데이터에서 weight_decay가 특히 중요
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Scheduler: CosineAnnealingLR ─────────────────────────
    # learning rate를 코사인 곡선으로 서서히 감소시켜
    # 학습 후반에 세밀한 수렴을 유도합니다.
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
    print("[ 학습 시작 — EfficientNet-B3 ]")
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

        # Early Stopping — 소규모 데이터에서 과도한 학습 방지
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
    print("\n[Test 평가] best_model_effb3.pth 가중치로 복원 중...")
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
