"""Compute confusion matrices for sign and grammar predictions."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from models.fusion import FusionModel
from train.dataset import BdSLDataset, SignerSplits


def parse_args():
    parser = argparse.ArgumentParser(description="Confusion matrix computation.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--train-signers", nargs="+", required=True)
    parser.add_argument("--val-signers", nargs="+", required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("confusion_matrices.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    signer_splits = SignerSplits(args.train_signers, args.val_signers, args.test_signers)
    dataset = BdSLDataset(args.manifest, args.landmarks, signer_splits, split="test")
    loader = DataLoader(dataset, batch_size=64)
    device = torch.device(args.device)
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    sign_true, sign_pred = [], []
    grammar_true, grammar_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sign_logits, grammar_logits = model(batch)
            sign_true.append(batch["sign_label"].cpu())
            sign_pred.append(sign_logits.argmax(dim=1).cpu())
            grammar_true.append(batch["grammar_label"].cpu())
            grammar_pred.append(grammar_logits.argmax(dim=1).cpu())

    sign_true = torch.cat(sign_true).numpy()
    sign_pred = torch.cat(sign_pred).numpy()
    grammar_true = torch.cat(grammar_true).numpy()
    grammar_pred = torch.cat(grammar_pred).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(confusion_matrix(sign_true, sign_pred), ax=axes[0], cmap="Blues", annot=False, cbar=True)
    axes[0].set_title("Sign Confusion Matrix")
    sns.heatmap(confusion_matrix(grammar_true, grammar_pred), ax=axes[1], cmap="Greens", annot=True, cbar=True)
    axes[1].set_title("Grammar Confusion Matrix")
    fig.tight_layout()
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
