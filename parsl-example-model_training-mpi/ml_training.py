
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import TensorDataset, DataLoader

# TRAINING_EPOCHS = 500
# RANDOM_STATE = 42

# class SingleNeuronNN(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.hidden = nn.Linear(20, 1)
#         self.output = nn.Linear(1, 1)

#     def forward(self, x):
#         x = torch.relu(self.hidden(x))
#         x = self.output(x)
#         return x


# def train_model(model_path):

#     torch.set_num_threads(1)

#     print("\n" + "=" * 65)
#     print("[STAGE 1] NEURAL NETWORK TRAINING")
#     print("=" * 65)

#     # Generate dataset
#     X, y = make_classification(
#         n_samples=5000,
#         n_features=20,
#         n_informative=15,
#         random_state=RANDOM_STATE
#     )

#     X_train, _, y_train, _ = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=RANDOM_STATE
#     )

#     # Normalize data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)

#     X_tensor = torch.tensor(
#         X_train,
#         dtype=torch.float32
#     )

#     y_tensor = torch.tensor(
#         y_train.reshape(-1, 1),
#         dtype=torch.float32
#     )

#     dataset = TensorDataset(X_tensor, y_tensor)

#     loader = DataLoader(
#         dataset,
#         batch_size=128,
#         shuffle=True
#     )

#     model = SingleNeuronNN()

#     criterion = nn.BCEWithLogitsLoss()

#     optimizer = optim.Adam(
#         model.parameters(),
#         lr=0.001
#     )

#     start_time = time.monotonic()

#     print("[ML TRAINING] Starting training...")

#     # Exactly 500 epochs
#     for epoch in range(1, TRAINING_EPOCHS + 1):

#         model.train()

#         total_loss = 0
#         correct = 0
#         total = 0

#         for batch_X, batch_y in loader:

#             optimizer.zero_grad()

#             predictions = model(batch_X)

#             loss = criterion(predictions, batch_y)

#             loss.backward()

#             optimizer.step()

#             total_loss += loss.item() * len(batch_y)

#             predicted_classes = (
#                 torch.sigmoid(predictions) >= 0.5
#             ).float()

#             correct += (
#                 predicted_classes == batch_y
#             ).sum().item()

#             total += len(batch_y)

#         avg_loss = total_loss / total
#         accuracy = correct / total

#         elapsed = time.monotonic() - start_time

#         print(
#             f"[EPOCH {epoch:04d}/{TRAINING_EPOCHS}] "
#             f"Loss: {avg_loss:.6f} | "
#             f"Accuracy: {accuracy:.4f} | "
#             f"Elapsed: {elapsed:.2f}s",
#             flush=True
#         )

#     training_time = time.monotonic() - start_time

#     # Save trained model
#     torch.save(
#         {
#             "model_state_dict": model.state_dict(),
#             "scaler_mean": torch.tensor(
#                 scaler.mean_,
#                 dtype=torch.float32
#             ),
#             "scaler_scale": torch.tensor(
#                 scaler.scale_,
#                 dtype=torch.float32
#             ),
#             "epochs": TRAINING_EPOCHS,
#             "training_time": training_time
#         },
#         model_path
#     )

#     print("\n[ML TRAINING] Training completed")
#     print("[ML TRAINING] Model saved:", model_path)

#     return model_path

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


TRAINING_EPOCHS = 500
RANDOM_STATE = 42


class SingleNeuronNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 1)
        self.output = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)


def train_model(model_path, log_path=None):
    torch.set_num_threads(1)

    log_file = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_file = open(log_path, "w", buffering=1)

    def log_print(*args, **kwargs):
        # Keep normal worker output, and also write a durable progress log.
        print(*args, **kwargs)
        if log_file is not None:
            file_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key not in ("file", "flush")
            }
            print(*args, file=log_file, flush=True, **file_kwargs)

    try:
        log_print("\n" + "=" * 65)
        log_print("[STAGE 1] NEURAL NETWORK TRAINING")
        log_print("=" * 65)

        X, y = make_classification(
            n_samples=5000,
            n_features=20,
            n_informative=15,
            random_state=RANDOM_STATE,
        )

        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(
            y_train.reshape(-1, 1),
            dtype=torch.float32,
        )

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
        )

        model = SingleNeuronNN()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        start_time = time.monotonic()
        log_print("[ML TRAINING] Starting training...")

        for epoch in range(1, TRAINING_EPOCHS + 1):
            model.train()

            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()

                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)

                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)

                predicted_classes = (
                    torch.sigmoid(predictions) >= 0.5
                ).float()

                correct += (
                    predicted_classes == batch_y
                ).sum().item()
                total += len(batch_y)

            avg_loss = total_loss / total
            accuracy = correct / total
            elapsed = time.monotonic() - start_time

            log_print(
                f"[EPOCH {epoch:04d}/{TRAINING_EPOCHS}] "
                f"Loss: {avg_loss:.6f} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Elapsed: {elapsed:.2f}s",
                flush=True,
            )

        training_time = time.monotonic() - start_time

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler_mean": torch.tensor(
                    scaler.mean_,
                    dtype=torch.float32,
                ),
                "scaler_scale": torch.tensor(
                    scaler.scale_,
                    dtype=torch.float32,
                ),
                "epochs": TRAINING_EPOCHS,
                "training_time": training_time,
            },
            model_path,
        )

        log_print("\n[ML TRAINING] Training completed")
        log_print("[ML TRAINING] Model saved:", model_path)

        return model_path

    finally:
        if log_file is not None:
            log_file.close()
