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
