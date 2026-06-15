#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ── configuration ─────────────────────────────────────────────

SEED = 42

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
DATA_DIR = CACHE_DIR / "data"
MODEL_DIR = CACHE_DIR / "models"
MODEL_PATH = MODEL_DIR / "vit_stl10.pt"

# image / patch-tokenizer settings
# STL-10 ships 96x96 images (9x the linear resolution of CIFAR-10's 32x32),
# so the displayed samples are sharp instead of blocky/pixelated.
IMG_SIZE = 96
PATCH_SIZE = 12
CHANNELS = 3
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 64
TOKEN_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS  # 432

# vision-transformer settings (trained from scratch)
NUM_CLASSES = 10
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FFN_DIM = 256
DROPOUT = 0.1

# training settings
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TEST_SIZE = 0.2
DEFAULT_NUM_INFERENCES = 3

# If True, retrain from scratch even if a cached model exists.
# (The CLI flag --retrain also turns this on for a single run.)
RETRAIN = False

# Default number of GPUs to use.  0 = CPU, 1 = single GPU, >1 = DataParallel.
# Overridden by the --numgpus CLI flag if provided.
NUM_GPUS = 1

# Normalisation constants used when loading STL-10.  Kept in module scope
# so we can de-normalise images for display.
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── patch tokenizer (the "tokenizer" for images) ─────────────

def tokenize_image_batch(images, patch_size):
    """
    Turn a batch of images into a sequence of patch tokens — the image
    equivalent of text tokenization.

    images: (N, C, H, W) tensor
    returns: (N, num_patches, patch_size*patch_size*C) tensor
    """
    n, c, h, w = images.shape
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # patches: (N, C, n_h, n_w, P, P) → (N, n_h, n_w, C, P, P)
    patches = patches.contiguous().permute(0, 2, 3, 1, 4, 5)
    # flatten each patch into a single token vector
    patches = patches.reshape(n, -1, c * patch_size * patch_size)
    return patches


# ── tokenized dataset wrapper ────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


# ── vision transformer (from scratch, no pretrained weights) ─

class VisionTransformer(nn.Module):
    """Small ViT trained from scratch on STL-10 patch tokens."""

    def __init__(self, token_dim, num_patches, embed_dim, num_heads,
                 num_layers, ffn_dim, num_classes, dropout):
        super().__init__()

        self.patch_embed = nn.Linear(token_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, tokens, labels=None):
        b = tokens.size(0)

        x = self.patch_embed(tokens)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        logits = self.head(cls_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return loss, logits


# ── training / evaluation loops ──────────────────────────────

def _reduce_loss(loss):
    """`nn.DataParallel` returns one loss per GPU; collapse to a scalar."""
    return loss.mean() if loss.dim() > 0 else loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total = 0.0, 0
    non_blocking = device.type == "cuda"
    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        loss, _ = model(tokens, labels)
        loss = _reduce_loss(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, total = 0.0, 0
    non_blocking = device.type == "cuda"
    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        loss, logits = model(tokens, labels)
        loss = _reduce_loss(loss)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        all_preds.extend(logits.argmax(-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, np.array(all_preds), np.array(all_labels)


# ── caching: dataset + model ─────────────────────────────────

def get_or_download_dataset():
    """Use cached STL-10 if it exists, otherwise download and cache it."""
    extracted_dir = DATA_DIR / "stl10_binary"

    if extracted_dir.exists():
        print(f"[CACHE] Dataset found at {DATA_DIR} — using cached copy.")
        download = False
    else:
        print(f"[DOWNLOAD] STL-10 not cached — downloading to {DATA_DIR}...")
        download = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return datasets.STL10(
        root=str(DATA_DIR), split="train", download=download, transform=transform,
    )


def build_model(device):
    return VisionTransformer(
        token_dim=TOKEN_DIM,
        num_patches=NUM_PATCHES,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=FFN_DIM,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ).to(device)


def maybe_load_model(device):
    """Return (model, was_cached).  Loads cached weights if available."""
    model = build_model(device)
    if MODEL_PATH.exists():
        print(f"[CACHE] Model found at {MODEL_PATH} — loading cached weights.")
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        return model, True
    print(f"[INIT] No cached model — built fresh model (will train and save to {MODEL_PATH}).")
    return model, False


def unwrap_model(model):
    """Return the underlying module, stripping any DataParallel wrapper."""
    return model.module if isinstance(model, nn.DataParallel) else model


def wrap_multi_gpu(model, num_gpus):
    """Wrap model in DataParallel across the first `num_gpus` GPUs."""
    if num_gpus > 1 and torch.cuda.device_count() >= num_gpus:
        device_ids = list(range(num_gpus))
        names = [torch.cuda.get_device_name(i) for i in device_ids]
        print(f"[MULTI-GPU] Wrapping model in DataParallel across {num_gpus} GPU(s):")
        for i, n in zip(device_ids, names):
            print(f"             [{i}] {n}")
        return nn.DataParallel(model, device_ids=device_ids)
    return model


def save_model(model):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(unwrap_model(model).state_dict(), MODEL_PATH)
    print(f"[SAVE] Model weights saved to {MODEL_PATH}")


# ── image display ────────────────────────────────────────────

def denormalize_image(img_tensor):
    """Reverse the Normalize transform so the image can be displayed."""
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def show_inference_images(images, infos):
    """Open a matplotlib window showing each test image with prediction info."""
    n = len(infos)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=150)
    if rows * cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).reshape(-1)

    for ax, img_tensor, info in zip(axes, images, infos):
        # "lanczos" resampling smooths the upscaled 96x96 image so it renders
        # cleanly instead of as blocky nearest-neighbour pixels.
        ax.imshow(denormalize_image(img_tensor), interpolation="lanczos")
        colour = "green" if info["correct"] else "red"
        ax.set_title(
            f"true: {info['true']}\n"
            f"pred: {info['pred']}  ({info['confidence']:.1%})",
            color=colour, fontsize=10,
        )
        ax.axis("off")

    for ax in axes[len(infos):]:
        ax.axis("off")

    fig.suptitle("Sample inferences on tokenized STL-10 test set", fontsize=12)
    fig.tight_layout()
    plt.show()


# ── main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vision Transformer (from scratch) trained on tokenized STL-10."
    )
    parser.add_argument(
        "-n", "--num-inferences", type=int, default=DEFAULT_NUM_INFERENCES,
        help=f"How many test inferences to display (default: {DEFAULT_NUM_INFERENCES})",
    )
    parser.add_argument(
        "--retrain", action="store_true", default=RETRAIN,
        help="Retrain the model on the tokenized dataset even if a cached "
             "model exists.  Defaults to the RETRAIN constant in this file.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not open the matplotlib window with the inference images.",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU even if a CUDA GPU is available.",
    )
    parser.add_argument(
        "--numgpus", type=int, default=NUM_GPUS,
        help=f"Number of GPUs to use.  0 = CPU.  >1 enables DataParallel.  "
             f"Defaults to the NUM_GPUS constant in this file (currently {NUM_GPUS}).",
    )
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── device selection ────────────────────────────────────
    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if args.cpu or args.numgpus == 0:
        device = torch.device("cpu")
        num_gpus = 0
        reason = "--cpu flag set" if args.cpu else "numgpus=0 requested"
        print(f"[DEVICE] {reason} — using CPU.\n")
    elif available_gpus == 0:
        device = torch.device("cpu")
        num_gpus = 0
        print("[DEVICE] No CUDA GPU detected — falling back to CPU.\n")
    else:
        requested = args.numgpus
        if requested > available_gpus:
            print(f"[WARN] Requested {requested} GPUs but only {available_gpus} available — clamping.")
            requested = available_gpus
        num_gpus = max(1, requested)

        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        print(f"[DEVICE] Detected {available_gpus} GPU(s); using {num_gpus}.")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            print(f"         [{i}] {name} ({vram:.1f} GB VRAM)")
        print(f"         CUDA {torch.version.cuda}  |  torch {torch.__version__}")
        print(f"         cudnn.benchmark enabled for speed.\n")
    pin_memory = device.type == "cuda"

    # 1. dataset (cached on disk)
    dataset = get_or_download_dataset()
    print(f"Loaded STL-10: {len(dataset)} images, {NUM_CLASSES} classes\n")

    # 2. tokenize every image with the patch tokenizer
    print("Tokenizing images into patch tokens...")
    images = torch.stack([img for img, _ in dataset])
    labels = np.array([lbl for _, lbl in dataset])
    tokens = tokenize_image_batch(images, PATCH_SIZE)
    print(f"  images shape : {tuple(images.shape)}")
    print(f"  tokens shape : {tuple(tokens.shape)} (N, num_patches={NUM_PATCHES}, token_dim={TOKEN_DIM})\n")

    # 3. split the TOKEN dataset into train / test
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=SEED, stratify=labels,
    )
    train_tokens, test_tokens = tokens[train_idx], tokens[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    print(f"Train: {len(train_labels)}   Test: {len(test_labels)}\n")

    train_ds = TokenDataset(train_tokens, train_labels)
    test_ds = TokenDataset(test_tokens, test_labels)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, pin_memory=pin_memory,
    )

    # 4. model (cached on disk)
    model, cached = maybe_load_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    model_device = next(model.parameters()).device
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model on        : {model_device}")
    model = wrap_multi_gpu(model, num_gpus)
    print()

    # 5. training (skipped when cached unless --retrain)
    if cached and not args.retrain:
        print("[SKIP] Skipping training — using cached model weights.\n")
    else:
        if cached and args.retrain:
            print("[RETRAIN] --retrain flag set — retraining from scratch.\n")
            model = build_model(device)
            model = wrap_multi_gpu(model, num_gpus)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print("=" * 60)
        print("TRAINING")
        print("=" * 60)
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_preds, val_true = evaluate(model, test_loader, device)
            val_acc = accuracy_score(val_true, val_preds)
            print(
                f"  Epoch {epoch:>2}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}"
            )
        save_model(model)

    # 6. final test-set evaluation
    test_loss, test_preds, test_true = evaluate(model, test_loader, device)
    acc = accuracy_score(test_true, test_preds)
    prec = precision_score(test_true, test_preds, average="weighted", zero_division=0)
    rec = recall_score(test_true, test_preds, average="weighted", zero_division=0)
    f1 = f1_score(test_true, test_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(test_true, test_preds)

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"  Test loss : {test_loss:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1 Score  : {f1:.4f}  (weighted)")
    print()
    print("  Confusion Matrix (rows=actual, cols=predicted):")
    header = "         " + "".join(f"{c[:5]:>6}" for c in STL10_CLASSES)
    print(header)
    for i, row in enumerate(cm):
        print(f"  {STL10_CLASSES[i][:5]:>6}" + "".join(f"{v:>6}" for v in row))
    print()
    print("  Classification Report:")
    print(classification_report(
        test_true, test_preds, target_names=STL10_CLASSES, zero_division=0,
    ))

    # 7. show N example inferences (default 3)
    n = max(1, args.num_inferences)
    n = min(n, len(test_labels))
    print("=" * 60)
    print(f"SAMPLE INFERENCES (n={n})")
    print("=" * 60)

    sample_idx = np.random.choice(len(test_labels), size=n, replace=False)
    display_images = []
    display_infos = []

    # Single-sample inferences don't benefit from DataParallel — use the
    # underlying module directly to avoid scatter/gather overhead.
    inference_model = unwrap_model(model)
    inference_model.eval()
    with torch.no_grad():
        for i, idx in enumerate(sample_idx, 1):
            tok = test_tokens[idx].unsqueeze(0).to(device)
            true_lbl = int(test_labels[idx])

            _, logits = inference_model(tok)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_lbl = int(np.argmax(probs))
            confidence = float(probs[pred_lbl])
            correct = pred_lbl == true_lbl
            mark = "OK" if correct else "MISS"

            top3 = np.argsort(probs)[::-1][:3]
            top_str = ", ".join(f"{STL10_CLASSES[j]} {probs[j]:.2%}" for j in top3)

            print(f"\n  [{i}/{n}] test index {idx}  [{mark}]")
            print(f"      Token shape : {tuple(test_tokens[idx].shape)}  (patches x patch_features)")
            print(f"      True label  : {STL10_CLASSES[true_lbl]}")
            print(f"      Predicted   : {STL10_CLASSES[pred_lbl]}  (confidence {confidence:.2%})")
            print(f"      Top-3       : {top_str}")

            display_images.append(images[test_idx[idx]])
            display_infos.append({
                "true": STL10_CLASSES[true_lbl],
                "pred": STL10_CLASSES[pred_lbl],
                "confidence": confidence,
                "correct": correct,
            })
    print()

    if not args.no_show:
        print("Opening matplotlib window with sample inferences...")
        show_inference_images(display_images, display_infos)


if __name__ == "__main__":
    main()
