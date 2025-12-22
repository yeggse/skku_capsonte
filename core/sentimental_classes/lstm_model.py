# core/sentimental_classes/lstm_model.py

from __future__ import annotations
import torch
import torch.nn as nn


class SentimentalLSTM(nn.Module):
    """
    SentimentalAgent에서 사용하는 기본 LSTM 모델.

    입력:  x.shape = (batch, seq_len, input_dim)
    출력:  (batch, 1)  # 다음날 수익률(return)을 예측한다고 가정
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)       # (B, T, H)
        out = out[:, -1, :]         # 마지막 타임스텝 (B, H)
        out = self.dropout(out)
        out = self.fc(out)          # (B, 1)  ← 예측된 수익률
        return out
