from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import SpectrumSmilesDataset, collate_fn, load_smiles
from model import SpectrumToSmilesModel
from tokenizer import build_tokenizer


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int):
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def make_mlm_inputs(y: torch.Tensor, tokenizer, mask_prob: float):
    masked = y.clone()
    valid = (y != tokenizer.pad_id) & (y != tokenizer.bos_id) & (y != tokenizer.eos_id)
    rand = torch.rand_like(y.float())
    mlm_mask = valid & (rand < mask_prob)
    masked[mlm_mask] = tokenizer.mask_id
    return masked, mlm_mask


def run_epoch(model, loader, optimizer, tokenizer, device, train: bool, mask_prob: float, lm_weight: float, mlm_weight: float):
    model.train(train)
    total_loss = 0.0
    total_lm = 0.0
    total_mlm = 0.0
    total_tokens = 0
    pbar = tqdm(loader, desc="train" if train else "val")

    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)  # (B,T)

        # CLM (causal next-token prediction)
        y_in = y[:, :-1]
        y_out = y[:, 1:]
        logits_lm = model(
            x,
            y_in,
            causal=True,
            tgt_key_padding_mask=(y_in == tokenizer.pad_id),
        )  # (B,T-1,V)
        lm_loss = F.cross_entropy(
            logits_lm.reshape(-1, logits_lm.size(-1)),
            y_out.reshape(-1),
            ignore_index=tokenizer.pad_id,
        )

        # MLM (bidirectional decoding over masked target sequence)
        y_masked, mlm_mask = make_mlm_inputs(y, tokenizer, mask_prob=mask_prob)
        logits_mlm = model(
            x,
            y_masked,
            causal=False,
            tgt_key_padding_mask=(y_masked == tokenizer.pad_id),
        )  # (B,T,V)
        if mlm_mask.any():
            mlm_loss = F.cross_entropy(logits_mlm[mlm_mask], y[mlm_mask])
        else:
            mlm_loss = torch.zeros((), dtype=lm_loss.dtype, device=lm_loss.device)

        loss = lm_weight * lm_loss + mlm_weight * mlm_loss

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        with torch.no_grad():
            non_pad = (y_out != tokenizer.pad_id).sum().item()
            total_loss += loss.item() * max(non_pad, 1)
            total_lm += lm_loss.item() * max(non_pad, 1)
            total_mlm += mlm_loss.item() * max(non_pad, 1)
            total_tokens += max(non_pad, 1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lm=f"{lm_loss.item():.4f}", mlm=f"{mlm_loss.item():.4f}")

    return (
        total_loss / max(total_tokens, 1),
        total_lm / max(total_tokens, 1),
        total_mlm / max(total_tokens, 1),
    )


@torch.no_grad()
def evaluate_token_acc(model, loader, tokenizer, device):
    model.eval()
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="eval_token")
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)
        y_in = y[:, :-1]
        y_out = y[:, 1:]
        logits = model(
            x,
            y_in,
            causal=True,
            tgt_key_padding_mask=(y_in == tokenizer.pad_id),
        )
        pred = logits.argmax(dim=-1)
        mask = (y_out != tokenizer.pad_id)
        correct += ((pred == y_out) & mask).sum().item()
        total += mask.sum().item()
        pbar.set_postfix(acc=f"{(correct / max(total, 1)):.4f}")
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_top1_exact_match(model, loader, tokenizer, device, max_len: int = 256):
    model.eval()
    total = 0
    match = 0
    pbar = tqdm(loader, desc="eval_top1")
    for x, y in pbar:
        x = x.to(device)
        pred_ids = model.greedy_decode(
            x,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_len,
        ).cpu().tolist()
        gt_ids = y.tolist()

        for p, g in zip(pred_ids, gt_ids):
            p_smiles = tokenizer.decode(p)
            g_smiles = tokenizer.decode(g)
            total += 1
            if p_smiles == g_smiles:
                match += 1
        pbar.set_postfix(acc=f"{(match / max(total, 1)):.4f}")

    return match / max(total, 1)


def main():
    parser = argparse.ArgumentParser("Train spectrum->SMILES elucidation model")
    parser.add_argument("--ir_csv", type=str, required=True)
    parser.add_argument("--raman_csv", type=str, required=True)
    parser.add_argument("--smiles_txt", type=str, required=True)
    parser.add_argument("--input_modality", type=str, default="ir", choices=["ir", "raman", "ir_raman"])

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_smiles_len", type=int, default=256)
    parser.add_argument("--mask_prob", type=float, default=0.45)
    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--mlm_weight", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_decoder_layers", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default="elucidation/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    smiles_list = load_smiles(args.smiles_txt)
    tokenizer = build_tokenizer(smiles_list)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    full_ds = SpectrumSmilesDataset(
        ir_csv=args.ir_csv,
        raman_csv=args.raman_csv,
        smiles_txt=args.smiles_txt,
        tokenizer=tokenizer,
        input_modality=args.input_modality,
        max_smiles_len=args.max_smiles_len,
    )
    train_idx, val_idx, test_idx = split_indices(len(full_ds), args.val_ratio, args.test_ratio, args.seed)
    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)
    test_ds = Subset(full_ds, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_id),
    )
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    model = SpectrumToSmilesModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_decoder_layers=args.num_decoder_layers,
        spectral_channel=2 if args.input_modality == "ir_raman" else 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_lm, tr_mlm = run_epoch(
            model, train_loader, optimizer, tokenizer, device, train=True,
            mask_prob=args.mask_prob, lm_weight=args.lm_weight, mlm_weight=args.mlm_weight
        )
        va_loss, va_lm, va_mlm = run_epoch(
            model, val_loader, optimizer, tokenizer, device, train=False,
            mask_prob=args.mask_prob, lm_weight=args.lm_weight, mlm_weight=args.mlm_weight
        )
        val_token_acc = evaluate_token_acc(model, val_loader, tokenizer, device)
        val_top1_acc = evaluate_top1_exact_match(
            model, val_loader, tokenizer, device, max_len=args.max_smiles_len
        )
        print(
            f"train_loss={tr_loss:.6f} (lm={tr_lm:.6f}, mlm={tr_mlm:.6f})  "
            f"val_loss={va_loss:.6f} (lm={va_lm:.6f}, mlm={va_mlm:.6f})  "
            f"val_token_acc={val_token_acc:.4f}  val_top1_acc={val_top1_acc:.4f}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "tokenizer": tokenizer.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, save_dir / "last.pt")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, save_dir / "best.pt")
            print(f"Saved best: {save_dir / 'best.pt'}")

    print("Training done.")
    test_top1_acc = evaluate_top1_exact_match(
        model, test_loader, tokenizer, device, max_len=args.max_smiles_len
    )
    print(f"Final test_top1_acc={test_top1_acc:.4f}")


if __name__ == "__main__":
    main()
