"""Run ablation studies for different modality combinations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.classifier import MultiTaskHead
from models.encoders import FaceEncoder, HandEncoder, PoseEncoder
from models.fusion import FusionMLP
from train.dataset import BdSLDataset, SignerSplits


class AblationModel(nn.Module):
    def __init__(self, use_hands: bool, use_face: bool, use_pose: bool):
        super().__init__()
        self.use_hands = use_hands
        self.use_face = use_face
        self.use_pose = use_pose
        self.hand_encoder = HandEncoder() if use_hands else None
        self.face_encoder = FaceEncoder() if use_face else None
        self.pose_encoder = PoseEncoder() if use_pose else None
        dims = []
        if self.hand_encoder:
            dims.append(self.hand_encoder.config.model_dim)
        if self.face_encoder:
            dims.append(self.face_encoder.config.model_dim)
        if self.pose_encoder:
            dims.append(self.pose_encoder.config.model_dim)
        self.fusion = FusionMLP(sum(dims), hidden_dims=(256, 128))
        self.head = MultiTaskHead(128)

    def forward(self, batch):
        feats = []
        if self.hand_encoder:
            hand = torch.cat((batch["hand_left"], batch["hand_right"]), dim=-1)
            feats.append(self.hand_encoder(hand.view(hand.size(0), hand.size(1), -1)))
        if self.face_encoder:
            feats.append(self.face_encoder(batch["face"].view(batch["face"].size(0), batch["face"].size(1), -1)))
        if self.pose_encoder:
            feats.append(self.pose_encoder(batch["pose"].view(batch["pose"].size(0), batch["pose"].size(1), -1)))
        fused = self.fusion(torch.cat(feats, dim=1))
        return self.head(fused)


def run_ablation(name: str, config: tuple[bool, bool, bool], train_loader, val_loader, device):
    model = AblationModel(*config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(5):
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad()
            sign_logits, grammar_logits = model(batch)
            loss = criterion(sign_logits, batch["sign_label"]) + 0.5 * criterion(
                grammar_logits, batch["grammar_label"]
            )
            loss.backward()
            optimizer.step()
    accuracy = evaluate(model, val_loader, device)
    print(f"{name}: accuracy={accuracy:.3f}")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            sign_logits, _ = model(batch)
            correct += (sign_logits.argmax(dim=1) == batch["sign_label"]).sum().item()
            total += batch["sign_label"].size(0)
    return correct / total


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation studies.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("landmarks", type=Path)
    parser.add_argument("--train-signers", nargs="+", required=True)
    parser.add_argument("--val-signers", nargs="+", required=True)
    parser.add_argument("--test-signers", nargs="+", required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    signer_splits = SignerSplits(args.train_signers, args.val_signers, args.test_signers)
    train_loader = DataLoader(
        BdSLDataset(args.manifest, args.landmarks, signer_splits, split="train"),
        batch_size=64,
        shuffle=True,
    )
    val_loader = DataLoader(
        BdSLDataset(args.manifest, args.landmarks, signer_splits, split="val"),
        batch_size=64,
        shuffle=False,
    )
    device = torch.device(args.device)

    ablations = {
        "hands-only": (True, False, False),
        "face-only": (False, True, False),
        "pose-only": (False, False, True),
        "hands+face": (True, True, False),
        "full": (True, True, True),
    }

    for name, cfg in ablations.items():
        run_ablation(name, cfg, train_loader, val_loader, device)


if __name__ == "__main__":
    main()
