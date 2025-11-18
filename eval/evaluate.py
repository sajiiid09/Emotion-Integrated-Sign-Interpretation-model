"""Evaluation script for BdSL models."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.classifier import MultiTaskHead
from models.encoders import FaceEncoder, HandEncoder, PoseEncoder
from models.fusion import FusionMLP
from train.dataset import BdSLDataset


class FusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_encoder = HandEncoder()
        self.face_encoder = FaceEncoder()
        self.pose_encoder = PoseEncoder()
        fusion_dim = self.hand_encoder.config.model_dim + self.face_encoder.config.model_dim + self.pose_encoder.config.model_dim
        self.fusion = FusionMLP(input_dim=fusion_dim)
        self.head = MultiTaskHead(128)

    def forward(self, batch):
        hand = torch.cat((batch["hand_left"], batch["hand_right"]), dim=-1)
        hand_feat = self.hand_encoder(hand.view(hand.size(0), hand.size(1), -1))
        face_feat = self.face_encoder(batch["face"].view(batch["face"].size(0), batch["face"].size(1), -1))
        pose_feat = self.pose_encoder(batch["pose"].view(batch["pose"].size(0), batch["pose"].size(1), -1))
        fused = torch.cat([hand_feat, face_feat, pose_feat], dim=1)
        fused = self.fusion(fused)
        return self.head(fused)


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
