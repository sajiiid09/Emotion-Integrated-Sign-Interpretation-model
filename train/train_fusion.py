"""Training script for multimodal fusion BdSL model."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.fusion import FusionModel
from train.dataset import BdSLDataset


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("train_fusion")


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion model.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
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


def train():
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

    model = FusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in loader_train:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            sign_logits, grammar_logits = model(batch)
            loss = criterion(sign_logits, batch["sign_label"]) + 0.5 * criterion(
                grammar_logits, batch["grammar_label"]
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch["sign_label"].size(0)
        scheduler.step()

        val_loss, val_acc = evaluate(model, loader_val, device, criterion)
        LOGGER.info(
            "Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.3f",
            epoch + 1,
            total_loss / len(loader_train.dataset),
            val_loss,
            val_acc,
        )

    torch.save(model.state_dict(), Path("fusion_model.pt"))


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sign_logits, grammar_logits = model(batch)
            loss = criterion(sign_logits, batch["sign_label"]) + 0.5 * criterion(
                grammar_logits, batch["grammar_label"]
            )
            total_loss += loss.item() * batch["sign_label"].size(0)
            correct += (sign_logits.argmax(dim=1) == batch["sign_label"]).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


if __name__ == "__main__":
    train()
