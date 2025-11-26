"""Training script for single-modality baselines."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.classifier import MultiTaskHead
from models.config import MODALITY_ENCODERS, MODALITY_TENSORS
from train.dataset import BdSLDataset


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_baselines")


class SingleStreamModel(nn.Module):
    def __init__(self, encoder: nn.Module, num_signs: int = 60, num_grammar: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = MultiTaskHead(encoder.config.model_dim, num_signs, num_grammar)

    def forward(self, inputs):
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        features = self.encoder(x)
        return self.head(features)


def run_epoch(model, loader, optimizer, device, modality, train: bool = True):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        inputs = MODALITY_TENSORS[modality](batch).to(device)
        sign_labels = batch["sign_label"].to(device)
        grammar_labels = batch["grammar_label"].to(device)
        if train:
            optimizer.zero_grad()
        sign_logits, grammar_logits = model(inputs)
        loss = criterion(sign_logits, sign_labels) + 0.5 * criterion(grammar_logits, grammar_labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct += (sign_logits.argmax(dim=1) == sign_labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("modality", choices=list(MODALITY_ENCODERS.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-signers", nargs="+", required=True)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument(
        "--no-pin-memory",
        action="store_false",
        dest="pin_memory",
        help="Disable DataLoader pin_memory (enabled by default for GPU training).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train_dataset = BdSLDataset(args.manifest, args.landmarks, args.train_signers, split="train")
    val_dataset = BdSLDataset(args.manifest, args.landmarks, args.train_signers, split="val")
    loader_train = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    loader_val = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )

    encoder = MODALITY_ENCODERS[args.modality]()
    model = SingleStreamModel(encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, loader_train, optimizer, device, args.modality, True)
        val_loss, val_acc = run_epoch(model, loader_val, optimizer, device, args.modality, False)
        LOGGER.info(
            "Epoch %d: train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch + 1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

    torch.save(model.state_dict(), Path(f"baseline_{args.modality}.pt"))


if __name__ == "__main__":
    main()
