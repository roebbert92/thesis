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
    "gaz_use_mentions": True,
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
#T5_ASP_LOWNERGAZ.update({
#    "adam_weight_decay": 0.011738749999999989,
#    "asp_dropout_rate": 0.4540625,
#    "asp_hidden_dim": 633,
#    "num_epochs": 16,
#    "plm_learning_rate": 0.00017496219281663535,
#    "search_algorithm": "bm25",
#    "search_topk": 12,
#    "task_learning_rate": 0.0035849253731343286,
#    "train_search_dropout": 0.05492957746478871,
#    "use_labels": True,
#    "use_mentions": True,
#    "warmup_ratio": 0.37917808219178084,
#    "name": "t5_asp_lownergaz"
#})
T5_ASP_LOWNERGAZ.update({
    "adam_weight_decay": 0.018862500000000015,
    "asp_dropout_rate": 0.43875,
    "asp_hidden_dim": 799,
    "num_epochs": 17,
    "plm_learning_rate": 0.00020887755102040807,
    "search_algorithm": "bm25",
    "search_topk": 12,
    "task_learning_rate": 0.003949473684210526,
    "train_search_dropout": 0.028260869565217374,
    "use_labels": True,
    "use_mentions": True,
    "warmup_ratio": 0.20864864864864865,
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
    "search_topk": 12,
    "task_learning_rate": 0.003949473684210526,
    "train_search_dropout": 0.028260869565217374,
    "use_labels": True,
    "use_mentions": True,
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
    "gaz_use_mentions": True,
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
    "search_topk": 6,
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

BEST_T5_ASP = T5_ASP.copy()
BEST_T5_ASP.update({
    "ckpt_path":
    "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/03_checkpoints/t5_asp/last.ckpt",
    "name": "best_t5_asp"
})

FULL_WNUT_T5_ASP_LOWNERGAZ_SENT = T5_ASP_LOWNERGAZ_SENT.copy()
FULL_WNUT_T5_ASP_LOWNERGAZ_SENT.update({
    "adam_weight_decay": 0.008067866817204384,
    "gaz_search_topk": 12,
    "num_epochs": 51,
    "plm_learning_rate": 0.00012035795477287743,
    "search_topk": 16,
    "sent_search_topk": 6,
    "task_learning_rate": 0.0074485651615560184,
    "name": "wnut_t5_asp_lownergaz_sent"
})

BEST_PRETRAINED_T5_ASP_LOWNERGAZ_SENT = T5_ASP_LOWNERGAZ_SENT.copy()
BEST_PRETRAINED_T5_ASP_LOWNERGAZ_SENT.update({
    "adam_weight_decay":
    0.01805954081911936,
    "asp_dropout_rate":
    0.22760117960387666,
    "gaz_search_topk":
    6,
    "plm_learning_rate":
    0.00025950985292715387,
    "sent_search_topk":
    5,
    "search_topk":
    11,
    "task_learning_rate":
    0.0021225113190029856,
    "train_search_dropout":
    0.029444504313170923,
    "warmup_ratio":
    0.36693271098631797,
    "num_epochs":
    15,
    "ckpt_path":
    "/home/loebbert/projects/thesis/experiments/01_performance/data/seed_1/03_checkpoints/t5_asp_lownergaz_sent/last.ckpt",
    "name":
    "best_wnut_t5_asp_lownergaz_sent"
})

WORST_PRETRAINED_T5_ASP_LOWNERGAZ_SENT = T5_ASP_LOWNERGAZ_SENT.copy()
WORST_PRETRAINED_T5_ASP_LOWNERGAZ_SENT.update({
    "adam_weight_decay":
    0.010792175245094616,
    "asp_dropout_rate":
    0.439294967901422,
    "gaz_search_topk":
    6,
    "num_epochs":
    18,
    "plm_learning_rate":
    0.00014868807960086362,
    "sent_search_topk":
    5,
    "search_topk":
    11,
    "task_learning_rate":
    0.0026384650282772083,
    "train_search_dropout":
    0.08316258683721132,
    "warmup_ratio":
    0.251818821694038,
    "ckpt_path":
    "/home/loebbert/projects/thesis/experiments/02_content/data/seed_2/03_checkpoints/size_4000/error_ratio_15/last.ckpt",
    "name":
    "worst_wnut_t5_asp_lownergaz_sent"
})
