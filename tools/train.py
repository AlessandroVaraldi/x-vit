#!/usr/bin/env python3
# =============================================================
#  train_tiny_vit_improved.py ― Tiny ViT (CIFAR‑100) w/ modern tricks
# -------------------------------------------------------------
#  * Longer cosine schedule with warm‑up
#  * RandAugment, MixUp & CutMix, RandomErasing
#  * Label‑smoothing cross‑entropy
#  * DropPath (stochastic depth)
#  * EMA (exponential moving average) of weights for eval
#  * Cosine LR with warm‑up, AdamW
#  * Exports vit_cifar100.onnx compatible with onnx2int8.py
# =============================================================

import argparse, math, os, random, time, copy, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Pillow ≥10 int-enum compat
try:
    from PIL import Image
    _BICUBIC = Image.Resampling.BICUBIC
except Exception:
    from PIL import Image
    _BICUBIC = Image.BICUBIC

# -------------------------------------------------------------
#  Utility – DropPath (stochastic depth)
# -------------------------------------------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # broadcast over dims
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarise
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# -------------------------------------------------------------
#  Core model – identical structural layout to previous version
# -------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm exposing gamma/beta."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class PatchEmbed(nn.Module):
    """Linear patch embedding (unfold + linear)"""
    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.weight = nn.Parameter(torch.randn(dim, in_chans * patch_size * patch_size) * 0.02)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        B, C, H, W = x.shape
        unfold = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        patches = unfold.transpose(1, 2)  # B × P × (C·p²)
        return F.linear(patches, self.weight, self.bias)  # B × P × dim


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class FeedForward(nn.Module):
    def __init__(self, dim, dff, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, dff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dff, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dff, drop_path_rate=0., dropout=0.):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, dff, dropout)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.ln1(x)))
        x = x + self.drop_path(self.ffn(self.ln2(x)))
        return x


class TinyViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=256, depth=8, heads=4, dff=512,
                 num_classes=100, drop_path_rate=0., dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # stochastic depth linear decay across layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dff, dpr[i], dropout) for i in range(depth)
        ])
        self.head = nn.Linear(dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, images):
        x = self.patch_embed(images)  # B × P × D
        cls = self.cls_token.expand(images.size(0), -1, -1)
        x = torch.cat([x, cls], dim=1)
        for blk in self.layers:
            x = blk(x)
        return self.head(x[:, -1])  # CLS token

# -------------------------------------------------------------
#  MixUp & CutMix helpers
# -------------------------------------------------------------

def mixup_cutmix_data(x, y, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    if np.random.rand() > prob:
        return x, y, y, 1.0  # no mix

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    if np.random.rand() < 0.5:
        # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x = lam * x + (1 - lam) * x[index]
    else:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        H, W = x.size(2), x.size(3)
        cut_w = int(W * math.sqrt(1 - lam))
        cut_h = int(H * math.sqrt(1 - lam))
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        x[:, :, max(cy - cut_h // 2, 0):min(cy + cut_h // 2, H),
              max(cx - cut_w // 2, 0):min(cx + cut_w // 2, W)] = \
            x[index, :, max(cy - cut_h // 2, 0):min(cy + cut_h // 2, H),
                      max(cx - cut_w // 2, 0):min(cx + cut_w // 2, W)]
        lam = 1 - (cut_w * cut_h) / (H * W)
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

# -------------------------------------------------------------
#  Scheduler with linear warm‑up + cosine decay
# -------------------------------------------------------------

def cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / optimizer.defaults['lr'], 0.5 * (1. + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------------------------------------------------
#  Utilities
# -------------------------------------------------------------

def load_pretrained(model, path, strict_head=False):
    """Carica pesi ImageNet e ignora il classifier se dimensioni diverse."""
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not strict_head:
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
    msg = model.load_state_dict(sd, strict=False)
    print(f"Loaded pretrained: {path} – missing={msg.missing_keys}, unexpected={msg.unexpected_keys}")

# -------------------------------------------------------------
#  Dataset helpers
# -------------------------------------------------------------

def build_imagenet_dataloaders(root, img_size, bs, workers=8):
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), interpolation=_BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        normalize,
    ])
    val_t = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=_BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    tr_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_t)
    val_ds = datasets.ImageFolder(os.path.join(root, "val"),   transform=val_t)
    tr_ld = DataLoader(tr_ds, batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=bs*2, shuffle=False, num_workers=workers)
    return tr_ld, val_ld

# -------------------------------------------------------------
#  Training routine (pretrain & finetune)
# -------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Seeds
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    # ---------------------------------------------------------
    #  Choose stage & dataloaders
    # ---------------------------------------------------------
    if args.stage == "pretrain":
        train_loader, val_loader = build_imagenet_dataloaders(
            args.imagenet, args.img_size, args.bs, workers=args.workers)
        num_classes = 1000
    else:  # finetune on CIFAR-100
        cifar_tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25),
            transforms.Normalize(mean=(0.5071, 0.4866, 0.4409),
                                 std=(0.2673, 0.2564, 0.2762)),
        ])
        cifar_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5071, 0.4866, 0.4409),
                                 std=(0.2673, 0.2564, 0.2762)),
        ])
        tr_ds = datasets.CIFAR100(root=args.data, train=True,  download=True, transform=cifar_tr)
        val_ds = datasets.CIFAR100(root=args.data, train=False, download=True, transform=cifar_val)
        train_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=args.bs*2, shuffle=False, num_workers=args.workers)
        num_classes = 100

    # Model
    model = TinyViT(img_size=args.img_size, patch_size=args.patch,
                    depth=args.layers, dim=args.dim, heads=args.heads,
                    dff=args.dff, num_classes=num_classes,
                    drop_path_rate=args.drop_path, dropout=args.dropout).to(device)

    if args.resume and args.stage == "finetune":
        load_pretrained(model, args.resume, strict_head=False)
    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # EMA model
    ema_model = copy.deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Optimiser & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = args.warmup * len(train_loader)
    scheduler = cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, args.min_lr)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.ls)

    best_acc = 0.0
    global_step = 0

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            t0 = time.time()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                # MixUp / CutMix
                imgs, y_a, y_b, lam = mixup_cutmix_data(imgs, labels, args.mixup_alpha, args.cutmix_alpha, args.mix_prob)

                optimizer.zero_grad()
                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                scheduler.step()
                global_step += 1

                # EMA update
                with torch.no_grad():
                    decay = args.ema
                    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.copy_(ema_p * decay + model_p * (1. - decay))

                running_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

            train_loss = running_loss / total
            train_acc = correct / total * 100

            # Validation with EMA weights
            ema_model.eval()
            correct = total_val = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = ema_model(imgs)
                    correct += (logits.argmax(1) == labels).sum().item()
                    total_val += imgs.size(0)
            val_acc = correct / total_val * 100

            elapsed = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:5.2f}% "
                f"val_acc={val_acc:5.2f}% lr={current_lr:.2e} time={elapsed:.1f}s")

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_dir = Path("checkpoints"); ckpt_dir.mkdir(exist_ok=True)
                tgt = ckpt_dir / ("best_tvit_imnet.pth" if args.stage=="pretrain" else "best_tvit_cifar.pth")
                torch.save(ema_model.state_dict(), tgt)
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        torch.save(ema_model.state_dict(), "checkpoints/interrupted_tvit.pth")
    finally:
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print("Training complete.")

    # Export best EMA model
    ema_model.eval()
    dummy = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(ema_model, dummy, "models/vit_cifar100.onnx",
                      input_names=["images"], output_names=["logits"],
                      opset_version=13, do_constant_folding=False)
    print("✅  ONNX model exported to models/vit_cifar100.onnx (EMA weights)")

    # ---------------------------------------------------------
    #  Export YAML config (onnx2int8 compat)
    # ---------------------------------------------------------
    try:
        import onnx, yaml, re
    except ImportError as e:
        print(f"⚠️  Skipping YAML export: {e}")
        return

    # --- infer main dims & depth from the freshly-saved ONNX ---
    onnx_model = onnx.load("models/vit_cifar100.onnx")
    d_model = d_ff = out_dim = None
    layers = 0
    qkv_re = re.compile(r"layers\.\d+\.attn\.qkv\.weight")
    ff1_re = re.compile(r"layers\.\d+\.ffn\.fc1\.weight")

    for init in onnx_model.graph.initializer:
        n = init.name
        if n.endswith("patch_embed.weight"):
            d_model = init.dims[0]
        elif ff1_re.fullmatch(n):
            d_ff = init.dims[0]
        elif qkv_re.fullmatch(n):
            layers += 1
        elif n == "head.weight":
            out_dim = init.dims[0]

    tokens    = model.patch_embed.n_patches + 1       # patches + CLS
    patches   = model.patch_embed.n_patches           # 64
    patch_dim = model.patch_embed.weight.shape[1]     # 48

    cfg = {
        "DMODEL":   d_model or args.dim,
        "DFF":      d_ff    or args.dff,
        "HEADS":    args.heads,
        "TOKENS":   tokens,
        "LAYERS":   layers,
        "OUT_DIM":  out_dim or 100,
        "EPS_SHIFT": 12,
        "shifts":   {},      # da riempire dopo la calibrazione INT8
        "PATCHES":  patches,
        "PATCH_DIM": patch_dim,
    }

    with open("models/vit_cifar100.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print("✅  YAML config exported to models/vit_cifar100.yaml")


# -------------------------------------------------------------
#  CLI
# -------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Tiny ViT on CIFAR‑100 with modern regularisation and export an onnx2int8‑ready model.")

    # Arguments for pretrain or finetune
    ap.add_argument("--data", default="./data", help="CIFAR-100 root directory")
    ap.add_argument("--imagenet", default="./imagenet", help="ImageNet-1k root (train/ val/ )")
    ap.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune", help="'pretrain' on ImageNet-1k or 'finetune' on CIFAR-100")

    # ImageNet pre-train args
    ap.add_argument("--img_size", type=int, default=224, help="input resolution for ImageNet pre-train")
    ap.add_argument("--patch", type=int, default=16, help="patch size (use 4 for CIFAR, 16 for ImageNet)")

    # Training hyper-params
    ap.add_argument("--epochs", type=int, default=200, help="total training epochs")
    ap.add_argument("--warmup", type=int, default=10, help="warm‑up epochs")
    ap.add_argument("--bs", type=int, default=512, help="batch size")
    ap.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
    ap.add_argument("--min_lr", type=float, default=1e-5, help="minimum LR for cosine schedule")
    ap.add_argument("--wd", type=float, default=5e-4, help="weight decay")

    # Model hyper‑params
    ap.add_argument("--dim", type=int, default=256, help="embedding dimension")
    ap.add_argument("--layers", type=int, default=8, help="number of Transformer blocks")
    ap.add_argument("--heads", type=int, default=4, help="attention heads")
    ap.add_argument("--dff", type=int, default=512, help="hidden units in FFN")

    # Regularisation
    ap.add_argument("--drop_path", type=float, default=0.1, help="DropPath rate")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout probability in FFN")
    ap.add_argument("--ls", type=float, default=0.1, help="label smoothing")

    # MixUp / CutMix
    ap.add_argument("--mixup_alpha", type=float, default=0.2, help="Beta α for MixUp")
    ap.add_argument("--cutmix_alpha", type=float, default=1.0, help="Beta α for CutMix")
    ap.add_argument("--mix_prob", type=float, default=0.5, help="probability of applying MixUp/CutMix")

    # EMA
    ap.add_argument("--ema", type=float, default=0.9999, help="EMA decay")

    # Misc
    ap.add_argument("--resume", help="checkpoint (.pth) to resume / load for fine-tune")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers")


    args = ap.parse_args()
    Path(args.data).mkdir(parents=True, exist_ok=True)
    train(args)

    # train.py --stage pretrain --imagenet /data/dataset/pytorch_only/imagenet/ --img_size 224 --patch 16 --epochs 300 --bs 512
    # train.py --stage finetune --data ./data --resume checkpoints/best_tvit_imnet.pth --img_size 32 --patch 4 --epochs 200 --bs 512
