from typing import Any, Dict
import torch
from transformers.generation.utils import GenerationMixin

from models.outputs import ASPSeq2SeqLMOutput


class ASPGenerationMixin(GenerationMixin):
    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.

        Returns:
            `bool`: Whether this model can generate sequences with `.generate()`.
        """
        # Detects whether `prepare_inputs_for_generation` has been overwritten, which is a requirement for generation
        # if "GenerationMixin" in str(self.prepare_inputs_for_generation):
        #     return False
        return True

    def prepare_inputs_for_generation(self,
                                      decoder_input_ids,
                                      past=None,
                                      attention_mask=None,
                                      head_mask=None,
                                      decoder_head_mask=None,
                                      cross_attn_head_mask=None,
                                      use_cache=None,
                                      encoder_outputs=None,
                                      decoder_encoder_input_ids=None,
                                      **kwargs):
        # cut decoder_input_ids if past is used
        if past is not None:
            cut_decoder_input_ids = decoder_input_ids[:, -1:]
        else:
            cut_decoder_input_ids = decoder_input_ids

        if "full_hidden_states" not in kwargs:
            kwargs["full_hidden_states"] = []  # initializing the list

        return {
            "input_ids":
            None,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,
            "decoder_input_ids":
            cut_decoder_input_ids,  # last decoder_input_ids
            "full_decoder_input_ids":
            decoder_input_ids,  # full_decoder_input_ids
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "encoder_input_ids": decoder_encoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "full_hidden_states": kwargs["full_hidden_states"],
            "decoder_pairing": kwargs["decoder_pairing"],
            "decoder_typing": kwargs["decoder_typing"],
            "use_cache": use_cache,
        }

    def _update_model_kwargs_for_generation(
            self,
            outputs: ASPSeq2SeqLMOutput,
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False) -> Dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        if "full_hidden_states" not in model_kwargs:
            model_kwargs["full_hidden_states"] = []

        model_kwargs["full_hidden_states"].append(
            outputs.decoder_hidden_states[-1].to(outputs.pairing.get_device()))
        model_kwargs["decoder_pairing"].append(outputs.pairing)
        model_kwargs["decoder_typing"].append(outputs.typing)

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat([
                    attention_mask,
                    attention_mask.new_ones((attention_mask.size(0), 1))
                ],
                                                           dim=-1)
        return model_kwargs

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx)
                for past_state in layer_past[:2]) + layer_past[2:], )
        return reordered_past