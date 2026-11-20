
# import torch
# import torch.nn as nn

# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score
# )


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


# def run_inference(model_path):

#     torch.set_num_threads(1)

#     print("\n" + "=" * 65)
#     print("[STAGE 3] NEURAL NETWORK INFERENCE")
#     print("=" * 65)

#     # Load trained model
#     checkpoint = torch.load(
#         model_path,
#         map_location="cpu",
#         weights_only=True
#     )

#     model = SingleNeuronNN()

#     model.load_state_dict(
#         checkpoint["model_state_dict"]
#     )

#     model.eval()

#     # Generate same dataset
#     X, y = make_classification(
#         n_samples=5000,
#         n_features=20,
#         n_informative=15,
#         random_state=RANDOM_STATE
#     )

#     _, X_test, _, y_test = train_test_split(
#         X,
#         y,
#         test_size=0.2,
#         random_state=RANDOM_STATE
#     )

#     # Apply training normalization
#     scaler_mean = checkpoint["scaler_mean"].numpy()
#     scaler_scale = checkpoint["scaler_scale"].numpy()

#     X_test = (X_test - scaler_mean) / scaler_scale

#     X_test_tensor = torch.tensor(
#         X_test,
#         dtype=torch.float32
#     )

#     print("[ML INFERENCE] Running predictions")

#     with torch.no_grad():

#         logits = model(X_test_tensor)

#         predictions = (
#             torch.sigmoid(logits) >= 0.5
#         ).int().numpy().flatten()

#     # Calculate metrics
#     metrics = {

#         "model": "Single Neuron Neural Network",

#         "hidden_neurons": 1,

#         "training_epochs": int(checkpoint["epochs"]),

#         "training_time_seconds": round(
#             float(checkpoint["training_time"]), 2
#         ),

#         "accuracy": round(
#             float(accuracy_score(y_test, predictions)), 4
#         ),

#         "precision": round(
#             float(precision_score(
#                 y_test, predictions, zero_division=0
#             )), 4
#         ),

#         "recall": round(
#             float(recall_score(
#                 y_test, predictions, zero_division=0
#             )), 4
#         ),

#         "f1_score": round(
#             float(f1_score(
#                 y_test, predictions, zero_division=0
#             )), 4
#         ),

#         "total_predictions": len(predictions)
#     }

#     print("\n[ML INFERENCE] RESULTS")

#     for key, value in metrics.items():
#         print(f"{key}: {value}")

#     return metrics

import torch
import torch.nn as nn

import sys
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

RANDOM_STATE = 42

class SingleNeuronNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = nn.Linear(20, 1)
        self.output = nn.Linear(1, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

def run_inference(model_path, log_path=None):

    # Limit PyTorch to one CPU thread
    torch.set_num_threads(1)

    # Prepare log file
    if log_path is not None:
        log_file_path = Path(log_path)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_file_path, "w", buffering=1)
    else:
        log_file = sys.stdout

    try:
        # Capture both standard output and errors
        with redirect_stdout(log_file), redirect_stderr(log_file):

            print("\n" + "=" * 65)
            print("[STAGE 3] NEURAL NETWORK INFERENCE")
            print("=" * 65)

            print(f"[ML INFERENCE] Model path: {model_path}")

            # Load trained model
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True
            )

            model = SingleNeuronNN()

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            model.eval()

            print("[ML INFERENCE] Trained model loaded successfully")

            # Generate the same dataset
            X, y = make_classification(
                n_samples=5000,
                n_features=20,
                n_informative=15,
                random_state=RANDOM_STATE
            )

            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE
            )

            # Apply training normalization
            scaler_mean = checkpoint["scaler_mean"].numpy()
            scaler_scale = checkpoint["scaler_scale"].numpy()

            X_test = (X_test - scaler_mean) / scaler_scale

            X_test_tensor = torch.tensor(
                X_test,
                dtype=torch.float32
            )

            print("[ML INFERENCE] Running predictions...")

            with torch.no_grad():

                logits = model(X_test_tensor)

                predictions = (
                    torch.sigmoid(logits) >= 0.5
                ).int().numpy().flatten()

            # Calculate metrics
            metrics = {
                "model": "Single Neuron Neural Network",
                "hidden_neurons": 1,
                "training_epochs": int(checkpoint["epochs"]),
                "training_time_seconds": round(
                    float(checkpoint["training_time"]), 2
                ),
                "accuracy": round(
                    float(accuracy_score(y_test, predictions)), 4
                ),
                "precision": round(
                    float(precision_score(
                        y_test, predictions, zero_division=0
                    )), 4
                ),
                "recall": round(
                    float(recall_score(
                        y_test, predictions, zero_division=0
                    )), 4
                ),
                "f1_score": round(
                    float(f1_score(
                        y_test, predictions, zero_division=0
                    )), 4
                ),
                "total_predictions": len(predictions)
            }

            print("\n[ML INFERENCE] RESULTS")
            print("-" * 65)

            for key, value in metrics.items():
                print(f"{key}: {value}")

            print("-" * 65)
            print("[STAGE 3] INFERENCE COMPLETED SUCCESSFULLY")
            print("=" * 65)

        return metrics

    finally:
        if log_path is not None:
            log_file.close()