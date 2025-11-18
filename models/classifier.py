"""Multi-task classifier heads for BdSL recognition."""
from __future__ import annotations

from torch import nn


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim: int, num_signs: int = 60, num_grammar: int = 3):
        super().__init__()
        self.sign_head = nn.Linear(input_dim, num_signs)
        self.grammar_head = nn.Linear(input_dim, num_grammar)

    def forward(self, features):
        sign_logits = self.sign_head(features)
        grammar_logits = self.grammar_head(features)
        return sign_logits, grammar_logits
