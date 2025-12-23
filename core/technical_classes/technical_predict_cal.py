import numpy as np
import torch
from config.agents_set import common_params

# MC Dropout / 표준편차 계산 / sigma clipping / confidence 계산
class TechnicalUncertaintyEstimator:
    def __init__(self, model):
        self.model = model

    def mc_dropout(self, X_tensor: torch.Tensor, n_samples: int):
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.model(X_tensor).cpu().numpy().flatten())
        return np.stack(preds)

    def estimate_sigma(self, preds: np.ndarray) -> float:
        std = np.std(preds, axis=0)
        sigma = float(std[-1])
        return max(sigma, common_params.get("sigma_min", 1e-6))
