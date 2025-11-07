from pathlib import Path
from typing import Any

import torch


class Checkpoint:
    state_dict: dict[str, Any]
    epoch: int
    test_acc: float
    avg_loss: float
    lr: float

    def __init__(self, state_dict: dict[str, Any], epoch: int, test_acc: float, avg_loss: float, lr: float):
        self.state_dict: dict[str, Any] = state_dict
        self.epoch: int = epoch
        self.test_acc: float = test_acc
        self.avg_loss: float = avg_loss
        self.lr: float = lr

    def __str__(self):
        return f"Checkpoint(epoch={self.epoch}, test_acc={self.test_acc:.4f}, avg_loss={self.avg_loss:.4f}, lr={self.lr:.2e})"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: Any) -> bool:
        return (
            self.state_dict == other.state_dict
            and self.epoch == other.epoch
            and self.test_acc == other.test_acc
            and self.avg_loss == other.avg_loss
            and self.lr == other.lr
        )

    def save(self, path: Path):
        save_dict = {
            "model_state_dict": self.state_dict,
            "epoch": self.epoch,
            "test_acc": self.test_acc,
            "avg_loss": self.avg_loss,
            "lr": self.lr,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: Path):
        save_dict = torch.load(path)
        state_dict = save_dict["model_state_dict"]
        epoch = save_dict["epoch"]
        test_acc = save_dict["test_acc"]
        avg_loss = save_dict["avg_loss"]
        lr = save_dict["lr"]

        return Checkpoint(state_dict, epoch, test_acc, avg_loss, lr)
