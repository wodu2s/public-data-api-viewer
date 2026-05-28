"""
X-ray 기반 질병 다중분류 (5-class) 학습 코드
모델: DenseNet121 (ImageNet pretrained)
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
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,   # AdamW weight decay → L2 정규화 효과
    "num_epochs": 50,

    # 데이터 분할 비율 (합이 1.0)
    "train_ratio": 0.70,
    "val_ratio":   0.15,
    "test_ratio":  0.15,

    # Scheduler 선택: "cosine" 또는 "plateau"
    "scheduler": "cosine",

    # Early stopping patience (None 이면 비활성화)
    "early_stopping_patience": 10,

    # 재현성을 위한 랜덤 시드
    "seed": 42,

    # 모델 저장 경로
    "save_path": "best_model.pth",

    # 그래프 저장 경로
    "plot_path": "training_history.png",

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

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121, DenseNet121_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# ============================================================
# 2. 랜덤 시드 고정 — 실험 재현성 확보
# ============================================================
def set_seed(seed: int) -> None:
    """모든 난수 생성기의 시드를 고정하여 실험 재현성을 보장합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN의 비결정적 알고리즘을 비활성화 (속도보다 재현성 우선)
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
    X-ray 이미지에 맞는 데이터 변환(transform)을 반환합니다.

    X-ray augmentation 주의사항:
    - 수직 뒤집기(Vertical Flip) 금지: 해부학적 위치(심장, 횡격막)가 역전되어
      임상적으로 불가능한 이미지를 생성합니다.
    - 회전 각도를 ±10도로 제한: 흉부 X-ray는 정면 촬영이 표준이므로
      과도한 회전은 병변 특징을 왜곡합니다.
    - ColorJitter를 약하게: 폐 음영 밀도가 진단의 핵심 단서이므로
      brightness/contrast 변화를 최소화합니다.
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet 통계값
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # X-ray는 좌우 대칭에 가까우므로 수평 뒤집기는 허용
        transforms.RandomHorizontalFlip(p=0.5),
        # 과도한 회전은 해부학적 왜곡 → ±10도로 제한
        transforms.RandomRotation(degrees=10),
        # 밝기/대비를 약하게만 변경 (폐 음영 과도 왜곡 방지)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Validation / Test: 증강 없이 크기 조정만 수행
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return {"train": train_transform, "val": val_test_transform, "test": val_test_transform}


# ============================================================
# 5. TransformDataset Wrapper — 데이터 누수 방지의 핵심
# ============================================================
class TransformDataset(Dataset):
    """
    Subset에 독립적인 transform을 적용하는 Wrapper 클래스.

    단순히 ImageFolder를 train/val/test로 나눠서 transform을 설정하면
    모든 Subset이 같은 transform을 공유합니다. 이 클래스를 사용하면
    각 Subset(train/val/test)에 서로 다른 transform을 안전하게 적용할 수 있습니다.
    """
    def __init__(self, subset: Subset, transform: transforms.Compose):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        # Subset에서 원본 이미지와 라벨을 가져옴
        image, label = self.subset.dataset[self.subset.indices[idx]]
        # transform이 설정되어 있으면 적용 (PIL 이미지 기준)
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 6. 데이터 분할 — Stratified Split으로 데이터 누수 방지
# ============================================================
def build_dataloaders(cfg: dict, transforms_dict: dict) -> tuple:
    """
    ImageFolder 형식의 데이터셋을 train/val/test로 분할하고
    각 DataLoader를 생성하여 반환합니다.

    [데이터 누수 방지 전략]
    1. transform이 없는 Raw ImageFolder를 먼저 로드합니다.
    2. 클래스별로 indices를 모아 stratified split을 수행합니다.
       → 각 클래스의 비율이 train/val/test 전반에 걸쳐 동일하게 유지됩니다.
    3. 분할된 indices로 Subset을 만들고, TransformDataset으로 각 transform을 적용합니다.
       → 동일한 이미지가 train과 val/test에 동시에 포함되는 일이 없습니다.
    """
    # transform 없이 로드 (split 후 TransformDataset에서 적용)
    raw_dataset = ImageFolder(root=cfg["data_dir"])

    print(f"\n[Dataset] 전체 이미지 수: {len(raw_dataset)}")
    print(f"[Dataset] 클래스 목록: {raw_dataset.classes}")
    for cls, idx in raw_dataset.class_to_idx.items():
        count = sum(1 for _, label in raw_dataset.samples if label == idx)
        print(f"          {cls}: {count}장")

    # 전체 indices와 라벨 추출
    all_indices = list(range(len(raw_dataset)))
    all_labels  = [label for _, label in raw_dataset.samples]

    # 1단계: 전체 → train 분리 (val+test 비율을 한 번에 계산)
    val_test_ratio = cfg["val_ratio"] + cfg["test_ratio"]
    train_idx, valtest_idx = train_test_split(
        all_indices,
        test_size=val_test_ratio,
        stratify=all_labels,
        random_state=cfg["seed"],
    )

    # 2단계: val+test 풀에서 val 과 test 분리
    valtest_labels = [all_labels[i] for i in valtest_idx]
    # val_ratio / val_test_ratio = val이 차지하는 비중
    val_ratio_within = cfg["val_ratio"] / val_test_ratio
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=(1.0 - val_ratio_within),
        stratify=valtest_labels,
        random_state=cfg["seed"],
    )

    print(f"\n[Split] Train: {len(train_idx)}장 | "
          f"Val: {len(val_idx)}장 | "
          f"Test: {len(test_idx)}장")

    # Subset + TransformDataset 생성
    train_ds = TransformDataset(Subset(raw_dataset, train_idx), transforms_dict["train"])
    val_ds   = TransformDataset(Subset(raw_dataset, val_idx),   transforms_dict["val"])
    test_ds  = TransformDataset(Subset(raw_dataset, test_idx),  transforms_dict["test"])

    # DataLoader 생성
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,                   # 학습 시 매 epoch마다 섞음
        num_workers=cfg["num_workers"],
        pin_memory=True,                # GPU 전송 속도 향상
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
# 7. 모델 빌드 — DenseNet121 Fine-tuning
# ============================================================
def build_model(num_classes: int) -> nn.Module:
    """
    DenseNet121 (ImageNet pretrained)의 마지막 classifier를
    num_classes 출력으로 교체한 모델을 반환합니다.

    [DenseNet121을 선택한 이유]
    - Dense Connectivity: 각 레이어가 이전 모든 레이어의 feature map을
      직접 연결(concatenate)하여 gradient가 깊은 레이어까지 원활하게 전달됩니다.
    - 파라미터 효율성: 약 7M 파라미터로 ResNet50(약 25M)보다 가볍습니다.
      7,500장의 소규모 데이터셋에서 과적합 위험을 낮출 수 있습니다.
    - 의료영상 검증: CheXNet(Stanford, 2017)에서 흉부 X-ray 14종 분류에
      DenseNet121을 사용해 방사선 전문의 수준 성능을 보고했습니다.
    - 내부 Dropout: DenseNet 블록 내부에 Dropout이 이미 포함되어 있어
      별도 레이어 추가 없이 정규화가 가능합니다.
    """
    # ImageNet pretrained 가중치 로드
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

    # DenseNet121의 원래 classifier: Linear(1024, 1000) [ImageNet 1000클래스]
    # in_features를 보존하고 out_features만 교체
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Linear(in_features, num_classes)

    print(f"\n[Model] DenseNet121 로드 완료 (pretrained=ImageNet)")
    print(f"[Model] Classifier: Linear({in_features}, {num_classes})")
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

        # 이전 gradient 초기화
        optimizer.zero_grad()

        # 순전파
        outputs = model(images)             # shape: (batch, num_classes)
        loss = criterion(outputs, labels)   # CrossEntropyLoss

        # 역전파 및 파라미터 업데이트
        loss.backward()
        optimizer.step()

        # 통계 누적
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

    @torch.no_grad() 데코레이터를 사용하여 gradient 계산을 비활성화,
    메모리 절약 및 속도 향상.

    Returns:
        avg_loss (float): 배치 평균 loss
        accuracy (float): 정확도 (0 ~ 1)
        all_preds (list): 배치 예측 결과
        all_labels (list): 실제 라벨
    """
    model.eval()  # Dropout 비활성화, BatchNorm 추론 모드

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
# 10. Test 최종 평가 — 지표 출력 및 시각화
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
    테스트 셋에 대한 최종 평가를 수행합니다.

    출력 지표:
    - Overall Accuracy
    - Per-class Precision, Recall, F1-score (classification_report)
    - Confusion Matrix (seaborn heatmap으로 저장)

    [Accuracy만 보면 안 되는 이유]
    클래스 불균형이 없더라도 의료 맥락에서는 클래스별 오류 비용이 다릅니다.
    - False Negative (환자를 정상으로 분류): 치료 기회 상실 → 비용 매우 높음
    - False Positive (정상을 환자로 분류): 불필요한 추가 검사 → 비용 상대적으로 낮음
    따라서 Recall(민감도)이 높아야 하며, Precision/F1/Confusion Matrix를
    함께 분석해야 모델의 실제 임상 유용성을 판단할 수 있습니다.
    """
    _, accuracy, all_preds, all_labels = evaluate(model, loader, criterion, device)

    print("\n" + "=" * 60)
    print("[ Test Set 최종 평가 결과 ]")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    # Per-class 지표 출력
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    ))

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)

    # Confusion Matrix 시각화
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
    ax.set_title("Confusion Matrix (Test Set)", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\n[저장] Confusion Matrix → {cm_path}")


# ============================================================
# 11. 학습 결과 그래프 출력
# ============================================================
def plot_training_history(history: dict, save_path: str) -> None:
    """
    epoch별 train/val loss와 accuracy 곡선을 한 그림에 저장합니다.

    Args:
        history: {"train_loss": [...], "val_loss": [...],
                  "train_acc": [...], "val_acc": [...]}
        save_path: 저장 경로 (예: "training_history.png")
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=15)

    # Loss 곡선
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", markersize=3)
    ax1.plot(epochs, history["val_loss"],   label="Val Loss",   marker="o", markersize=3)
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy 곡선
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
    # CrossEntropyLoss를 사용하는 이유:
    # 이 태스크는 "단일 라벨 다중분류" (각 X-ray는 하나의 진단)입니다.
    # CrossEntropyLoss는 내부적으로 Softmax를 적용하여 각 클래스에 대한
    # 확률 분포를 계산하고 정답 클래스의 로그 확률을 최대화합니다.
    # BCEWithLogitsLoss는 각 클래스를 독립적인 이진 분류로 취급하므로
    # 한 샘플에 여러 정답이 있는 "다중 라벨" 분류에 적합합니다.
    criterion = nn.CrossEntropyLoss()

    # ── Optimizer (AdamW + weight decay로 과적합 방지) ────────
    # weight_decay는 파라미터에 L2 패널티를 부여하여 과도한 가중치 증가를 억제합니다.
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    # ── Scheduler ────────────────────────────────────────────
    if cfg["scheduler"] == "cosine":
        # CosineAnnealingLR: learning rate를 코사인 곡선으로 서서히 감소
        # 끝에서 급격한 수렴을 유도하여 local minima를 탈출하는 데 유리
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["num_epochs"], eta_min=1e-6
        )
    else:
        # ReduceLROnPlateau: val_loss가 개선되지 않으면 lr을 절반으로 감소
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    # ── 학습 기록 초기화 ──────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }

    best_val_loss   = float("inf")
    best_model_wts  = copy.deepcopy(model.state_dict())
    patience_count  = 0
    patience_limit  = cfg["early_stopping_patience"]

    print("\n" + "=" * 60)
    print("[ 학습 시작 ]")
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

        # Scheduler 업데이트
        if cfg["scheduler"] == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # 기록 저장
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # 현재 learning rate 확인
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch:>3}/{cfg['num_epochs']}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.2e}"
        )

        # Best 모델 저장 — val_loss가 개선될 때만 저장 (과거 best 기준)
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, cfg["save_path"])
            print(f"  → Best model 저장 (val_loss: {best_val_loss:.4f})")
            patience_count = 0
        else:
            patience_count += 1

        # Early Stopping
        if patience_limit and patience_count >= patience_limit:
            print(f"\n[Early Stopping] {patience_limit} epochs 동안 개선 없음. 학습 종료.")
            break

    print("\n[학습 완료]")
    print(f"Best Val Loss: {best_val_loss:.4f}")

    # ── 그래프 저장 ──────────────────────────────────────────
    plot_training_history(history, cfg["plot_path"])

    # ── 최종 평가 (best model 가중치 복원 후 test) ─────────────
    print("\n[Test 평가] best_model.pth 가중치로 복원 중...")
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
