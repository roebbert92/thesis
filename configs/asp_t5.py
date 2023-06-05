from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class ASPT5Config:
    plm_pretrained_name_or_path: str
    plm_tokenizer_name: str
    model_max_length: int
    mention_start_token: str
    mention_end_token: str
    asp_hidden_dim: int
    asp_dropout_rate: float
    asp_init_std: float
    asp_activation: str
    num_labels: int
    max_nest_depth: int
    beam_size: int
    plm_learning_rate: float
    plm_scheduler: str
    task_learning_rate: float
    task_scheduler: str
    adam_eps: float
    adam_weight_decay: float
    warmup_ratio: float
    num_epochs: int
    gradient_accumulation_steps: int
    batch_size: int
    train_len: Optional[int] = 0
    fused: Optional[bool] = None


T5_BASE = asdict(
    ASPT5Config(plm_pretrained_name_or_path="t5-base",
                plm_tokenizer_name="t5-small",
                model_max_length=4096,
                mention_start_token="<m>",
                mention_end_token="</m>",
                asp_hidden_dim=150,
                asp_dropout_rate=0.3,
                asp_init_std=0.02,
                asp_activation="relu",
                num_labels=6,
                max_nest_depth=1,
                beam_size=1,
                plm_learning_rate=5e-5,
                plm_scheduler="linear_with_warmup",
                task_learning_rate=3e-4,
                task_scheduler="linear_with_warmup",
                adam_eps=1e-8,
                adam_weight_decay=0.1,
                warmup_ratio=0.05,
                num_epochs=20,
                batch_size=40,
                gradient_accumulation_steps=1))

T5_ASP_LOWNERGAZ_SENT = T5_BASE.copy()
T5_ASP_LOWNERGAZ_SENT.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "gaz_search_algorithm": "bm25",
    "gaz_search_topk": 6,
    "gaz_use_labels": True,
    "gaz_use_mentions": False,
    "num_epochs": 16,
    "plm_learning_rate": 0.00017496219281663535,
    "search_join_method": "reciprocal_rank_fusion",
    "search_topk": 8,
    "sent_search_algorithm": "ann",
    "sent_search_topk": 6,
    "sent_use_labels": True,
    "sent_use_mentions": True,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_lownergaz_sent"
})

T5_ASP_LOWNERGAZ = T5_BASE.copy()
T5_ASP_LOWNERGAZ.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "num_epochs": 16,
    "plm_learning_rate": 0.00017496219281663535,
    "search_algorithm": "bm25",
    "search_topk": 8,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "use_labels": True,
    "use_mentions": False,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_lownergaz"
})

T5_ASP_GAZ = T5_BASE.copy()
T5_ASP_GAZ.update({
    "adam_weight_decay": 0.018862500000000015,
    "asp_dropout_rate": 0.43875,
    "asp_hidden_dim": 799,
    "num_epochs": 17,
    "plm_learning_rate": 0.00020887755102040807,
    "search_algorithm": "bm25",
    "search_topk": 6,
    "task_learning_rate": 0.003949473684210526,
    "train_search_dropout": 0.028260869565217374,
    "use_labels": True,
    "use_mentions": False,
    "warmup_ratio": 0.20864864864864865,
    "name": "t5_asp_gaz"
})

T5_ASP_GAZ_SENT = T5_BASE.copy()
T5_ASP_GAZ_SENT.update({
    "adam_weight_decay": 0.011738749999999989,
    "asp_dropout_rate": 0.4540625,
    "asp_hidden_dim": 633,
    "gaz_search_algorithm": "bm25",
    "gaz_search_topk": 6,
    "gaz_use_labels": True,
    "gaz_use_mentions": False,
    "num_epochs": 24,
    "plm_learning_rate": 0.00017496219281663535,
    "search_join_method": "reciprocal_rank_fusion",
    "search_topk": 8,
    "sent_search_algorithm": "ann",
    "sent_search_topk": 6,
    "sent_use_labels": True,
    "sent_use_mentions": True,
    "task_learning_rate": 0.0035849253731343286,
    "train_search_dropout": 0.05492957746478871,
    "warmup_ratio": 0.37917808219178084,
    "name": "t5_asp_gaz_sent"
})

T5_ASP_SENT = T5_BASE.copy()
T5_ASP_SENT.update({
    "adam_weight_decay": 0.49637507889057786,
    "asp_dropout_rate": 0.3,
    "asp_hidden_dim": 142,
    "num_epochs": 20,
    "plm_learning_rate": 5e-05,
    "search_algorithm": "ann",
    "search_topk": 8,
    "task_learning_rate": 0.0013480523331922776,
    "train_search_dropout": 0.21126587935893093,
    "use_labels": True,
    "use_mentions": True,
    "warmup_ratio": 0.184451637360714,
    "name": "t5_asp_sent"
})

T5_ASP = T5_BASE.copy()
T5_ASP.update({
    "adam_weight_decay": 0.12402083333333332,
    "asp_dropout_rate": 0.11718749999999999,
    "asp_hidden_dim": 342,
    "num_epochs": 21,
    "plm_learning_rate": 0.00010693877551020426,
    "task_learning_rate": 0.00413396694214876,
    "warmup_ratio": 0.29414201183431954,
    "name": "t5_asp"
})