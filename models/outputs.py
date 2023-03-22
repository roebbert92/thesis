from transformers.modeling_outputs import Seq2SeqLMOutput
from dataclasses import dataclass
import torch
from typing import Optional, List

@dataclass
class ASPSeq2SeqLMOutput(Seq2SeqLMOutput):
    """
    Extra sequence-to-sequence language models outputs.
    Args:
        pairing (`torch.LongTensor` of shape `(1,)`, *optional*):
            pairing left brackets for right brackets.
        linking (`torch.LongTensor` of shape `(1,)`, *optional*):
            linking right brackets to right brackets.
        typing (`torch.LongTensor` of shape `(1,)`, *optional*):
            typing right brackets.
    """ 
    pairing: Optional[List[torch.LongTensor]] = None
    linking: Optional[List[torch.LongTensor]] = None
    typing: Optional[List[torch.LongTensor]] = None 
    