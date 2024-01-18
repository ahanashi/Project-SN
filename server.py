import os
import shutil
import torch
import pickle
import flwr as fl
from collections import OrderedDict
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from functools import partial
from torch.nn import BatchNorm2d

def always_true(*args, **kwargs):
    return True


results = {
    "mAP": [],
    "recall": [],
    "precision": [],
    "loss": [],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {
    "imgsz": 480,
    "batch": 16,
    "epochs": 1,
    "plots": True,
    "save": True,
    "mosaic": 0,
    "lr0": 1e-9,
    "device": device,
    "cache": False,
    "exist_ok": True,
    "model": "yolov8n.pt",
    "data": f"data/clients/test/cfg.yaml",
    "project": f"runs/test",
}


server = DetectionTrainer(overrides=kwargs)
server.setup_model()
server.model.is_fused = always_true
validator = DetectionValidator(
    dataloader=server.get_dataloader("data/clients/test/val", batch_size=16, rank=-1, mode="val"), args=kwargs
)


def get_eval_fn():
    """Return an evaluation function for server-side evaluation."""

    @torch.inference_mode()
    def evaluate(parameters):
        """Use validation set to evaluate the parameters and return a score."""

        server.model.eval()
        params_dict = zip(server.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        server.model.load_state_dict(state_dict, strict=False)

        r = validator(model=server.model)
        results["mAP"].append(r["metrics/mAP50(B)"])
        results["recall"].append(r["metrics/recall(B)"])
        results["precision"].append(r["metrics/precision(B)"])
        with open(f"results/test.pkl", "wb") as f:
            pickle.dump(results, f)

        return 100.0, {"mAP": r["metrics/mAP50(B)"]}

    return evaluate


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    shutil.rmtree("results", ignore_errors=True)
    shutil.rmtree("runs", ignore_errors=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("runs/test/train", exist_ok=True)

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=3,
        min_available_clients=3,
        eval_fn=get_eval_fn(),
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config= {"num_rounds": 20, "round_timeout": None}, #fl.server.ServerConfig(num_rounds=20, round_timeout=None),
        strategy=strategy,
    )
