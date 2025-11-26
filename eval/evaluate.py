"""Evaluation script for BdSL models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.fusion import FusionModel
from train.dataset import BdSLDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fusion model.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-signers", nargs="+", required=True)
    return parser.parse_args()


def evaluate_model():
    args = parse_args()
    device = torch.device(args.device)
    test_dataset = BdSLDataset(args.manifest, args.landmarks, args.train_signers, split="test")
    loader = DataLoader(test_dataset, batch_size=64)
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    correct_sign = 0
    correct_grammar = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sign_logits, grammar_logits = model(batch)
            correct_sign += (sign_logits.argmax(dim=1) == batch["sign_label"]).sum().item()
            correct_grammar += (grammar_logits.argmax(dim=1) == batch["grammar_label"]).sum().item()
            total += batch["sign_label"].size(0)

    print(f"Sign accuracy: {correct_sign / total:.3f}")
    print(f"Grammar accuracy: {correct_grammar / total:.3f}")


if __name__ == "__main__":
    evaluate_model()
