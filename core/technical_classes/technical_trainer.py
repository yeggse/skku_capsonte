import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from config.agents_set import agents_info, common_params


class TechnicalTrainer:
    def __init__(self, agent):
        self.agent = agent
        self.model = agent
        self.scaler = agent.scaler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_dataset(self, df_raw):
        cfg = agents_info[self.agent.agent_id]
        feature_cols = cfg["data_cols"]
        window = self.agent.window_size

        X = df_raw[feature_cols].values.astype(np.float32)
        close = df_raw["Close"].values
        y = (close[1:] / close[:-1] - 1.0).reshape(-1, 1)
        X = X[:-1]

        X_seq, y_seq = self.agent._create_sequences(X, y, window)

        y_scale = common_params.get("y_scale_factor", 100.0)
        y_seq *= y_scale

        self.scaler.fit_scalers(X_seq, y_seq)
        self.scaler.save(self.agent.ticker)

        X_scaled, y_scaled = self.scaler.transform(X_seq, y_seq)

        X_tensor = torch.tensor(X_scaled).float()
        y_tensor = torch.tensor(y_scaled).float().view(-1, 1)

        return X_tensor, y_tensor

    def fit(self, X, y):
        cfg = agents_info[self.agent.agent_id]
        batch_size = cfg["batch_size"]
        epochs = cfg["epochs"]

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg["learning_rate"])
        loss_fn = torch.nn.HuberLoss(
            delta=common_params.get("huber_loss_delta", 1.0)
        )

        for epoch in range(epochs):
            total = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f"[Train] Epoch {epoch+1}/{epochs} | Loss {total/len(loader):.6f}")

    def save(self):
        path = os.path.join(
            self.agent.model_dir,
            f"{self.agent.ticker}_{self.agent.agent_id}.pt"
        )
        torch.save({"model_state_dict": self.model.state_dict()}, path)
