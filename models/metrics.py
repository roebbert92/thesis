from torchmetrics import Metric
import torch
from collections import defaultdict


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

    def update(self, preds: list, targets: list):
        self.tp += torch.tensor(len(set(preds) & set(targets)),
                                dtype=torch.float)
        self.fn += torch.tensor(len(set(targets) - set(preds)),
                                dtype=torch.float)
        self.fp += torch.tensor(len(set(preds) - set(targets)),
                                dtype=torch.float)

    def compute(self):
        precision = self.tp / (self.fp + self.tp)
        recall = self.tp / (self.fn + self.tp)
        return (2 * precision * recall) / (precision + recall)


class FalsePositivesASP():
    def __init__(self) -> None:
        self.false_positives = defaultdict(list)

    def update(self, doc_id, preds: list, targets: list):
        for item in set(preds) - set(targets):
            self.false_positives[doc_id].append(item)

    def compute(self):
        return self.false_positives

    def reset(self):
        self.false_positives.clear()
