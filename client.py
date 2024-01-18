import os

# os.environ["YOLO_VERBOSE"] = "false"
from ultralytics import YOLO
from ultralytics.models.yolo.model import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

import torch
import pickle
import argparse
import flwr as fl
import pandas as pd
from glob import glob
from collections import OrderedDict
from functools import partial
from torch.nn import BatchNorm2d


def always_true(*args, **kwargs):
    return True


class Client(fl.client.NumPyClient):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        kwargs = {
            "imgsz": 480,
            "batch": 16,
            "epochs": 1,
            "plots": False,
            "save": False,
            "mosaic": 0,
            "lr0": 1e-9,
            "device": device,
            "cache": False,
            "exist_ok": True,
            "model": "yolov8n.pt",
            "data": f"data/clients/{idx}/cfg.yaml",
            "project": f"runs/{idx}",
        }
        self.client = DetectionTrainer(overrides=kwargs)
        self.client.setup_model()
        self.client.model.is_fused = always_true

        self.results = {
            "mAP": [],
            "recall": [],
            "precision": [],
            "loss": [],
        }
        self.train_num_samples = len(glob(f"data/clients/{idx}/train/*.jpg"))
        self.val_num_samples = len(glob(f"data/clients/{idx}/val/*.jpg"))

    def get_parameters(self):
        self.client.model.eval()
        return [val.cpu().numpy() for _, val in self.client.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.client.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.client.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.client.train()
        if os.path.exists(f"runs/{self.idx}.pkl"):
            with open(f"runs/{self.idx}.pkl", "rb") as f:
                self.results = pickle.load(f)
        r = self.client.validator.metrics
        self.results["mAP"].append(r.results_dict["metrics/mAP50(B)"])
        self.results["recall"].append(r.results_dict["metrics/recall(B)"])
        self.results["precision"].append(r.results_dict["metrics/precision(B)"])
        df = pd.read_csv(f"runs/{self.idx}/train/results.csv")
        loss = (
            7.5 * df["         train/cls_loss"].values[-1]
            + 0.5 * df["         train/box_loss"].values[-1]
            + 1.5 * df["         train/dfl_loss"].values[-1]
        )
        self.results["loss"].append(loss)
        with open(f"results/{self.idx}.pkl", "wb") as f:
            pickle.dump(self.results, f)
        return self.get_parameters(), self.train_num_samples, {}

    def evaluate(self, parameters, config):
        if not self.client.validator:
            return float("nan"), self.val_num_samples, {}
        return float(self.results["loss"][-1]), self.val_num_samples, {"mAP": float(self.results["mAP"][-1])}


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--client_idx", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    fl.client.start_numpy_client(server_address="localhost:8080", client=Client(args.client_idx))
