from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import SpectrumSmilesDataset, collate_fn
from model import SpectrumToSmilesModel
from tokenizer import SmilesTokenizer


def main():
    parser = argparse.ArgumentParser("Inference for spectrum->SMILES model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--ir_csv", type=str, required=True)
    parser.add_argument("--raman_csv", type=str, required=True)
    parser.add_argument("--input_modality", type=str, default="ir", choices=["ir", "raman", "ir_raman"])
    parser.add_argument("--smiles_txt", type=str, default=None, help="Optional ground-truth smiles for quick comparison")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--output_txt", type=str, default="elucidation/pred_smiles.txt")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    ckpt = torch.load(args.checkpoint, map_location=device)
    tok = SmilesTokenizer.from_state_dict(ckpt["tokenizer"])
    model_args = ckpt["args"]

    model = SpectrumToSmilesModel(
        vocab_size=tok.vocab_size,
        d_model=model_args["d_model"],
        nhead=model_args["nhead"],
        num_decoder_layers=model_args["num_decoder_layers"],
        spectral_channel=2 if args.input_modality == "ir_raman" else 1,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # smiles_txt is optional here; if not provided, use a dummy file with empty labels is not supported.
    # we pass checkpoint-time smiles path if available, otherwise require user input.
    smiles_txt = args.smiles_txt if args.smiles_txt is not None else model_args["smiles_txt"]
    ds = SpectrumSmilesDataset(
        ir_csv=args.ir_csv,
        raman_csv=args.raman_csv,
        smiles_txt=smiles_txt,
        tokenizer=tok,
        input_modality=args.input_modality,
        max_smiles_len=args.max_len,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tok.pad_id))

    preds = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Decoding"):
            x = x.to(device)
            out_ids = model.greedy_decode(x, bos_id=tok.bos_id, eos_id=tok.eos_id, max_len=args.max_len)
            out_ids = out_ids.cpu().tolist()
            for ids in out_ids:
                preds.append(tok.decode(ids))

    output = Path(args.output_txt)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for s in preds:
            f.write(s + "\n")
    print(f"Saved predictions: {output}")


if __name__ == "__main__":
    main()
