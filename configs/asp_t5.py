from dataclasses import dataclass


@dataclass
class ASP_T5_Config:
    plm_pretrained_name_or_path: str
    plm_tokenizer_name: str
    mention_start_token: str
    mention_end_token: str
    asp_hidden_dim: int
    asp_dropout_rate: float
    asp_init_std: float
    asp_activation: str
    num_labels: int
    max_nest_depth: int
    beam_size: int
    
    num_epochs: int
    # gradient_accumulation_steps: int
    batch_size: int
    train_len: int

T5_SMALL = ASP_T5_Config(
    plm_pretrained_name_or_path="t5-small",
    plm_tokenizer_name="t5-small",
    mention_start_token="<m>",
    mention_end_token="</m>",
)
