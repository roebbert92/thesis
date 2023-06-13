from typing import List
from torchmetrics import Metric
import torch
from collections import defaultdict
import pandas as pd


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


class ASPMetrics(Metric):
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
        self.add_state("error_type1",
                       default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx="sum")
        self.add_state("error_type2",
                       default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx="sum")
        self.add_state("error_type3",
                       default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx="sum")
        self.add_state("error_type4",
                       default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx="sum")
        self.add_state("error_type5",
                       default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx="sum")
        self.predictions = {}

    def update(self, id: str, preds: list, targets: list):
        preds_set = set(preds)
        targets_set = set(targets)
        tp = torch.tensor(len(preds_set & targets_set), dtype=torch.float)
        fn = torch.tensor(len(targets_set - preds_set), dtype=torch.float)
        fp = torch.tensor(len(preds_set - targets_set), dtype=torch.float)
        error_type1 = torch.tensor(len(
            ASPMetrics.get_error_type1(preds_set, targets_set)),
                                   dtype=torch.int)
        error_type2 = torch.tensor(len(
            ASPMetrics.get_error_type2(preds_set, targets_set)),
                                   dtype=torch.int)
        error_type3 = torch.tensor(len(
            ASPMetrics.get_error_type3(preds_set, targets_set)),
                                   dtype=torch.int)
        error_type4 = torch.tensor(len(
            ASPMetrics.get_error_type4(preds_set, targets_set)),
                                   dtype=torch.int)
        error_type5 = torch.tensor(len(
            ASPMetrics.get_error_type5(preds_set, targets_set)),
                                   dtype=torch.int)

        self.predictions[id] = preds
        # global update
        self.tp += tp
        self.fn += fn
        self.fp += fp
        self.error_type1 += error_type1
        self.error_type2 += error_type2
        self.error_type3 += error_type3
        self.error_type4 += error_type4
        self.error_type5 += error_type5

    @staticmethod
    def get_error_type1(preds: set, targets: set):
        # complete false positives - no overlaps
        false_positives = preds - targets
        complete_false_positives = set()
        sorted_targets = sorted(targets)
        for false_positive in sorted(false_positives):
            # check if overlap in targets
            overlap = False
            for target in sorted_targets:
                if target[1] <= false_positive[0]:
                    continue
                if target[0] >= false_positive[1]:
                    break
                if false_positive[1] >= target[0]:
                    overlap = True
            if not overlap:
                complete_false_positives.add(false_positive)
        return complete_false_positives

    @staticmethod
    def get_error_type2(preds: set, targets: set):
        # complete false negative - no overlaps
        false_negatives = targets - preds
        complete_false_negatives = set()
        sorted_preds = sorted(preds)
        for false_negative in sorted(false_negatives):
            # check if overlap in targets
            overlap = False
            for pred in sorted_preds:
                if pred[1] <= false_negative[0]:
                    continue
                if pred[0] >= false_negative[1]:
                    break
                if false_negative[1] >= pred[0]:
                    overlap = True
            if not overlap:
                complete_false_negatives.add(false_negative)
        return complete_false_negatives

    @staticmethod
    def get_error_type3(preds: set, targets: set):
        # wrong label, right span
        type3s = set()
        sorted_targets = sorted(targets)
        for pred in sorted(preds):
            for target in sorted_targets:
                if target[0] >= pred[1]:
                    break
                if target[1] <= pred[0]:
                    continue
                if pred[0] == target[0] and pred[1] == target[
                        1] and pred[2] != target[2]:
                    type3s.add(pred)
        return type3s

    @staticmethod
    def get_error_type4(preds: set, targets: set):
        # wrong label, overlapping span
        type4s = set()
        sorted_targets = sorted(targets)
        for pred in sorted(preds):
            for target in sorted_targets:
                if target[0] >= pred[1]:
                    break
                if target[1] <= pred[0]:
                    continue
                if (abs(pred[0] - target[0]) > 0 or
                        abs(pred[1] - target[1]) > 0) and pred[2] != target[2]:
                    type4s.add(pred)
        return type4s

    @staticmethod
    def get_error_type5(preds: set, targets: set):
        # right label, overlapping span
        type5s = set()
        sorted_targets = sorted(targets)
        for pred in sorted(preds):
            for target in sorted_targets:
                if target[0] >= pred[1]:
                    break
                if target[1] <= pred[0]:
                    continue
                if (abs(pred[0] - target[0]) > 0 or
                        abs(pred[1] - target[1]) > 0) and pred[2] == target[2]:
                    type5s.add(pred)
        return type5s

    @staticmethod
    def calc_precision(tp: torch.Tensor, fp: torch.Tensor):
        if not tp.is_nonzero() and not fp.is_nonzero():
            return torch.tensor(0.0, dtype=torch.float)
        return torch.round(tp / (fp + tp), decimals=4)

    @staticmethod
    def calc_recall(tp: torch.Tensor, fn: torch.Tensor):
        if not tp.is_nonzero() and not fn.is_nonzero():
            return torch.tensor(0.0, dtype=torch.float)
        return torch.round(tp / (fn + tp), decimals=4)

    @staticmethod
    def calc_f1(precision: torch.Tensor, recall: torch.Tensor):
        if not precision.is_nonzero() and not recall.is_nonzero():
            return torch.tensor(0.0, dtype=torch.float)
        return torch.round((2 * precision * recall) / (precision + recall),
                           decimals=4)

    def compute(self):
        return self.f1()

    def precision(self):
        return ASPMetrics.calc_precision(self.tp, self.fp)

    def recall(self):
        return ASPMetrics.calc_recall(self.tp, self.fn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        return ASPMetrics.calc_f1(precision, recall)

    def errors(self):
        return (self.error_type1.item(), self.error_type2.item(),
                self.error_type3.item(), self.error_type4.item(),
                self.error_type5.item())

    def metrics_per_sample(self, dataset: List[dict], types: List[str]):
        # dataframe with tp, fp, fn, error types per sample_id and entity type
        metrics = []
        for doc in dataset:
            doc_id = doc["doc_id"]
            assert doc_id in self.predictions
            for t in types:
                preds_set = set([
                    pred for pred in self.predictions[doc_id] if pred[2] == t
                ])
                targets_set = set([(ent["start"], ent["end"], ent["type"])
                                   for ent in doc["entities"]
                                   if ent["type"] == t])
                tp = len(preds_set & targets_set)
                fn = len(targets_set - preds_set)
                fp = len(preds_set - targets_set)
                error_type1 = len(
                    ASPMetrics.get_error_type1(preds_set, targets_set))
                error_type2 = len(
                    ASPMetrics.get_error_type2(preds_set, targets_set))
                error_type3 = len(
                    ASPMetrics.get_error_type3(preds_set, targets_set))
                error_type4 = len(
                    ASPMetrics.get_error_type4(preds_set, targets_set))
                error_type5 = len(
                    ASPMetrics.get_error_type5(preds_set, targets_set))
                metrics.append({
                    "doc_id": doc_id,
                    "targets": len(targets_set),
                    "entity_type": t,
                    "tp": tp,
                    "fn": fn,
                    "fp": fp,
                    "error_type1": error_type1,
                    "error_type2": error_type2,
                    "error_type3": error_type3,
                    "error_type4": error_type4,
                    "error_type5": error_type5,
                })
        return pd.DataFrame.from_records(metrics)

    def reset(self):
        super().reset()
        self.predictions.clear()


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
