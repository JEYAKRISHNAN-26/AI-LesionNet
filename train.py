"""Train AI-LesionNet on real, labelled NIfTI-derived case volumes."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from ailesionnet import AILesionNet, segmentation_loss
from ailesionnet.data import LesionDataset


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = (torch.sigmoid(model(x)) >= 0.5).float()
            dims = (1, 2, 3, 4)
            num = (2 * (pred * y).sum(dim=dims))
            den = pred.sum(dim=dims) + y.sum(dim=dims)
            scores.extend(torch.where(den > 0, num / den.clamp(min=1), torch.ones_like(den)).cpu().tolist())
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description='AI-LesionNet training')
    parser.add_argument('--manifest', required=True, help='Preprocessed manifest from preprocess.py')
    parser.add_argument('--output', default='runs/ai-lesionnet')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8, help='Manuscript setting; reduce if VRAM is insufficient')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--min-learning-rate', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='Unspecified in paper; default 0 for reproducibility')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--clip-norm', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--transformer-dropout', type=float, default=0.1)
    parser.add_argument('--dice-weight', type=float, default=0.5)
    parser.add_argument('--disable-fusion', action='store_true',
                        help='CNN-only ablation: omit transformer and modality cross-attention')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    seed_all(args.seed)
    device = torch.device(args.device)
    config = dict(base_channels=args.base_channels, embed_dim=args.embed_dim,
                  heads=args.heads, layers=args.layers, patch=args.patch,
                  dropout=args.dropout, transformer_dropout=args.transformer_dropout,
                  use_fusion=not args.disable_fusion)
    train_set = LesionDataset(args.manifest, 'train', augment=True)
    val_set = LesionDataset(args.manifest, 'val', augment=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_set, batch_size=1, num_workers=args.num_workers)
    model = AILesionNet(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    warm = max(0, args.warmup_epochs)
    def lr_multiplier(epoch):
        if warm and epoch < warm:
            return max((epoch + 1) / warm, 1e-4)
        import math
        progress = min(1.0, max(0.0, (epoch - warm) / max(1, args.epochs - warm)))
        floor = args.min_learning_rate / args.learning_rate
        return floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    # AMP speeds CUDA without changing full-precision CPU execution.
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'config.json').write_text(json.dumps(vars(args), indent=2))
    best, waiting = -1., 0
    for epoch in range(args.epochs):
        model.train()
        running, n = 0., 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = segmentation_loss(model(x), y, dice_weight=args.dice_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach().item() * len(x)
            n += len(x)
        validation = evaluate(model, val_loader, device)
        scheduler.step()
        log = {'epoch': epoch + 1, 'train_loss': running / max(n, 1),
               'val_dice': validation, 'learning_rate': optimizer.param_groups[0]['lr']}
        print(json.dumps(log), flush=True)
        with (dest / 'history.jsonl').open('a', encoding='utf8') as f:
            f.write(json.dumps(log) + '\n')
        if validation > best:
            best, waiting = validation, 0
            torch.save({'state_dict': model.state_dict(), 'model_config': config,
                        'epoch': epoch+1, 'val_dice': best}, dest / 'best.pt')
        else:
            waiting += 1
        if waiting >= args.patience:
            print('Early stopping reached')
            break
    print(f'Training complete; best val Dice={best:.5f}; model={dest / "best.pt"}')


if __name__ == '__main__':
    main()
