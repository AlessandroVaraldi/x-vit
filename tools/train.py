import argparse, math, os, random, time, copy, re
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import amp
from torch.optim.swa_utils import (AveragedModel, get_ema_multi_avg_fn, update_bn)
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as T          # >=0.19

torch.backends.cudnn.benchmark = True              # autotuner

# --------------------------- DropPath ------------------------
def drop_path(x, p: float = 0., training: bool = False):
    if p == 0. or not training: return x
    keep = 1 - p
    mask = keep + torch.rand((x.shape[0],) + (1,)*(x.ndim-1), dtype=x.dtype, device=x.device)
    return x.div(keep)*mask.floor()

class DropPath(nn.Module):
    def __init__(self, p=0.): 
        super().__init__()
        self.p = p

    def forward(self,x): return drop_path(x,self.p,self.training)

# --------------------------- Model --------------------------
class LayerNorm(nn.Module):
    def __init__(self,d,eps=1e-5):
        super().__init__()
        self.g=nn.Parameter(torch.ones(d))
        self.b=nn.Parameter(torch.zeros(d))
        self.eps=eps

    def forward(self,x):
        m, v = x.mean(-1, keepdim=True), x.var(-1, unbiased=False, keepdim=True)
        return self.g * (x - m) / torch.sqrt(v + self.eps) + self.b

class PatchEmbed(nn.Module):
    def __init__(self, img, patch, in_c=3, d=256):
        super().__init__()
        self.proj = nn.Conv2d(in_c, d, patch, patch)  # kernel=stride=patch
        self.np = (img // patch) ** 2

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)    # (B,N,d)
        return x

class MHSA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, D//self.h).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(y.transpose(1,2).reshape(B, T, D))

class FFN(nn.Module):
    def __init__(self, d, dff, p=0.): 
        super().__init__()
        self.fc1 = nn.Linear(d, dff)
        self.fc2 = nn.Linear(dff, d)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self,x):
        return self.fc2(self.drop(self.act(self.fc1(x))))

class Block(nn.Module):
    def __init__(self, d, h, dff, dp=0., do=0.):
        super().__init__()
        self.ln1, self.attn, self.ln2, self.ffn = LayerNorm(d), MHSA(d, h), LayerNorm(d), FFN(d, dff, do)
        self.dp = DropPath(dp) if dp > 0 else nn.Identity()

    def forward(self, x): 
        x = x + self.dp(self.attn(self.ln1(x)))
        return x + self.dp(self.ffn(self.ln2(x)))

class TinyViT(nn.Module):
    def __init__(self, img, patch, d, depth, heads, dff, classes, dp=0., do=0.):
        super().__init__()
        self.embed = PatchEmbed(img, patch, 3, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        dpr = torch.linspace(0, dp, depth).tolist()
        self.layers = nn.ModuleList([Block(d, heads, dff, dpr[i], do) for i in range(depth)])
        self.head = nn.Linear(d, classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.embed(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], 1)
        for blk in self.layers:
            x = blk(x)
        return self.head(x[:, 0])

# ---------------------- MixUp (tensor) ----------------------
def mixup(x, y, alpha=0.2, p=0.5):
    if torch.rand(1, device=x.device) > p:
        return x, y, y, 1.
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

# --------------- Cosine scheduler w/ warm‑up ---------------
def cosine_warmup(opt, warm, total, min_lr=1e-5):
    def fn(step):
        if step < warm:
            return step / warm
        prog = (step - warm) / (total - warm)
        return max(min_lr / opt.defaults['lr'], 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

# -------------------- DataLoaders helpers ------------------
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CIFAR_MEAN, CIFAR_STD = (0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)

def imagenet_loaders(root, img, bs, workers):
    train_t = T.Compose([
        T.RandomResizedCrop(img, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandAugment(),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    val_t = T.Compose([
        T.Resize(int(img * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(IMNET_MEAN, IMNET_STD),
    ])
    tr = datasets.ImageFolder(os.path.join(root, "train"), transform=train_t)
    te = datasets.ImageFolder(os.path.join(root, "val"), transform=val_t)
    tr_ld = DataLoader(tr, bs, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    te_ld = DataLoader(te, bs * 2, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True)
    return tr_ld, te_ld

def cifar_loaders(root, bs, workers):
    train_t = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.RandAugment(),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        T.RandomErasing(p=0.25)
    ])
    val_t = T.Compose([T.PILToTensor(), T.ToDtype(torch.float32, scale=True), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    tr = datasets.CIFAR100(root, True, download=True, transform=train_t)
    te = datasets.CIFAR100(root, False, download=True, transform=val_t)
    tr_ld = DataLoader(tr, bs, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True, prefetch_factor=4)
    te_ld = DataLoader(te, bs * 2, shuffle=False, num_workers=workers, pin_memory=True, persistent_workers=True)
    return tr_ld, te_ld

# --------------- Load pretrained (ignore head) --------------
def load_pretrained(model, path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    head_w = head_b = model.head.weight.shape[0]
    sd = {k: v for k, v in sd.items() if not (k.startswith("head.") and v.shape[0] != head_w)}
    msg = model.load_state_dict(sd, strict=False)
    print(f"Loaded {path}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")

# ------------------------- Train ---------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    random.seed(0); np.random.seed(0)
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)

    if args.stage == "pretrain":
        # ImageNet-1k
        ld_train, ld_val = imagenet_loaders(
            args.imagenet, args.img_size, args.bs, args.workers)
        num_classes = 1000
    else:
        # CIFAR-100
        args.img_size, args.patch = 32, 4
        ld_train, ld_val = cifar_loaders(
            args.data, args.bs, args.workers)
        num_classes = 100

    net = TinyViT(args.img_size, args.patch, args.dim, args.layers, args.heads, args.dff, num_classes, args.drop_path, args.dropout).to(device)

    if args.resume:
        load_pretrained(net, args.resume)

    ema = AveragedModel(net, device=device, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

    model = torch.compile(net, mode="max-autotune") if args.compile else net

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = amp.GradScaler()
    total  = len(ld_train) * args.epochs
    warm   = args.warmup * len(ld_train)
    sched  = cosine_warmup(opt, warm, total, args.min_lr)
    crit   = nn.CrossEntropyLoss(label_smoothing=args.ls)

    best = 0.

    # --------------------- Print model summary ---------------------
    PARAMS = [p for p in model.parameters()]

    tot_elems = sum(p.numel() for p in PARAMS)
    print(f"Parameters: {tot_elems:,}")

    def to_mib(bytes_): return bytes_ / 1024**2

    fp32_bytes = sum(p.numel() * 4 for p in PARAMS)
    print(f"Memory fp32 (ass.): {to_mib(fp32_bytes):.2f} MiB")

    dtype_bytes = sum(p.numel() * p.element_size() for p in PARAMS)
    print(f"Memory actual dtype: {to_mib(dtype_bytes):.2f} MiB")

    int8_bytes = sum(p.numel() * 1 for p in PARAMS)
    print(f"Memory int8 (quant): {to_mib(int8_bytes):.2f} MiB")

    print("Model compiled:", args.compile)
    print("Using mixed precision:", torch.cuda.is_available())

    try:
        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            loss_sum = correct = tot = 0
            for imgs, lbls in ld_train:
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                imgs, y1, y2, lam = mixup(imgs, lbls, args.mixup_alpha, args.mix_prob)
                opt.zero_grad(set_to_none=True)
                with amp.autocast(device_type='cuda'):
                    logits = model(imgs)
                    loss = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                ema.update_parameters(model)
                sched.step()
                step += 1

                loss_sum += loss.item() * imgs.size(0)
                # Only count accuracy for unmixed samples (lam == 1)
                if lam == 1.0:
                    correct += (logits.argmax(1) == y1).sum().item()
                    tot += imgs.size(0)
                # Otherwise, skip accuracy computation for mixed samples

            train_loss, train_acc = loss_sum / tot, 100 * correct / tot
            ema.eval()
            correct = tot_val = 0
            with torch.cuda.amp.autocast(), torch.no_grad():
                for imgs, lbl in ld_val:
                    imgs, lbl = imgs.to(device, non_blocking=True), lbl.to(device, non_blocking=True)
                    out = ema(imgs)
                    correct += (out.argmax(1) == lbl).sum().item()
                    tot_val += imgs.size(0)
            val_acc = 100 * correct / tot_val
            print(f"[E{epoch:03d}] loss {train_loss:.4f} acc {train_acc:5.2f}% "
                  f"val {val_acc:5.2f}% lr {sched.get_last_lr()[0]:.2e} "
                  f"time {time.time()-t0:.1f}s")
            if val_acc > best:
                best = val_acc
    except KeyboardInterrupt:
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(ema.module.state_dict(), "checkpoints/interrupted.pth")
        print("Interrupted, model saved.")
    finally: 
        print(f"Best val acc: {best:.2f}%")
        print("Interrupted, model saved.")

    # ----------- Export ONNX + YAML ----------------
    export_dir = Path("models")
    export_dir.mkdir(exist_ok=True)
    onnx_path = str(export_dir/"vit_model.onnx")
    base = ema.module if isinstance(ema, AveragedModel) else ema
    base.eval()
    dummy = torch.randn(1,3,args.img_size,args.img_size).to(device)
    torch.onnx.export(base, dummy, onnx_path,
                     input_names=["images"], output_names=["logits"],
                     opset_version=17,
                     do_constant_folding=False)

    print("ONNX exported ➜", onnx_path)

    try:
        import onnx,yaml
        onx = onnx.load(onnx_path)
        d_model = d_ff = out_dim = layers = None
        qkv = re.compile(r"layers\.\d+\.attn\.qkv\.weight")
        ff1 = re.compile(r"layers\.\d+\.ffn\.fc1\.weight")
        for init in onx.graph.initializer:
            n = init.name
            if n.endswith("patch_embed.w"):
                d_model = init.dims[0]
            elif ff1.fullmatch(n):
                d_ff = init.dims[0]
            elif qkv.fullmatch(n):
                layers = (layers or 0) + 1
            elif n == "head.weight":
                out_dim = init.dims[0]
        # Set defaults if values are still None
        if d_ff is None:
            d_ff = args.dff
        if out_dim is None:
            out_dim = num_classes
        if layers is None:
            layers = args.layers
        cfg = dict(DMODEL=d_model, 
                   DFF=d_ff, 
                   HEADS=args.heads,
                   TOKENS=base.embed.np + 1, 
                   LAYERS=layers, 
                   OUT_DIM=out_dim,
                   EPS_SHIFT=12, 
                   shifts={}, 
                   PATCHES=base.embed.np, 
                   PATCH_DIM=base.embed.proj.weight.shape[0])
        with open(export_dir / "vit_config.yaml", "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        print("YAML exported ➜", export_dir / "vit_config.yaml")
    except Exception as e:
        print("Skip YAML export:", e)

# --------------------------- CLI ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser("Tiny ViT train (ImageNet➜CIFAR)")
    p.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune")
    p.add_argument("--data", default="./data", help="root CIFAR-100")
    p.add_argument("--imagenet", default="./imagenet", help="root ImageNet train/ val/")
    p.add_argument("--img_size", type=int, default=224, help="input res (pretrain)")
    p.add_argument("--patch", type=int, default=16, help="patch size (pretrain)")
    # train
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--wd", type=float, default=5e-4)
    # model
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dff", type=int, default=512)
    # reg
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ls", type=float, default=0.1)
    # mixup
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--mix_prob", type=float, default=0.5)
    # ema
    p.add_argument("--ema", type=float, default=0.9999)
    # misc
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count()-2))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", help="checkpoint .pth to resume / fine-tune")
    args = p.parse_args()

    # set defaults for CIFAR stage
    if args.stage == "finetune":
        args.img_size, args.patch = 32, 4
    Path(args.data).mkdir(parents=True, exist_ok=True)
    train(args)

# python3 tools/train.py --stage pretrain --imagenet /data/dataset/pytorch_only/imagenet/ --bs 512 --epochs 300 --compile
# python3 tools/train.py --stage finetune  --data ./cifar-100-python --resume checkpoints/best_tvit_imnet.pth --bs 512 --epochs 200 --compile
