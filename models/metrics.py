from torchmetrics import Metric
import torch


class F1ASP(Metric):
    higher_is_better = True
    is_differentiable = False
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("tp",
                       default=torch.tensor(1e-6, dtype=torch.float),
                       dist_reduce_fx="sum")
        self.add_state("fn",
                       default=torch.tensor(1e-6, dtype=torch.float),
                       dist_reduce_fx="sum")
        self.add_state("fp",
                       default=torch.tensor(1e-6, dtype=torch.float),
                       dist_reduce_fx="sum")

    def update(self, preds: list, target: list):
        self.tp += torch.tensor(len(set(preds) & set(target)),
                                dtype=torch.float)
        self.fn += torch.tensor(len(set(target) - set(preds)),
                                dtype=torch.float)
        self.fp += torch.tensor(len(set(preds) - set(target)),
                                dtype=torch.float)

    def compute(self):
        precision = self.tp / (self.fp + self.tp)
        recall = self.tp / (self.fn + self.tp)
        return (2 * precision * recall) / (precision + recall)
