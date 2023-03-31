from typing import Any, List, Optional
import lightning.pytorch as pl
from transformers import T5Tokenizer, T5Model, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers.generation.utils import GenerationConfig
import torch
import torch.nn as nn
from models.outputs import ASPSeq2SeqLMOutput

import models.utils as utils
import models.generation as gen
from models.metrics import F1ASP

NEGINF = -20000


def get_scheduler_lambda(scheduler_type, warmup_steps, total_steps):
    if scheduler_type == 'linear_with_warmup':

        def lambda_rule(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - step) /
                float(max(1, total_steps - warmup_steps)))

        return lambda_rule
    elif scheduler_type == 'constant':
        return lambda _: 1.0
    elif scheduler_type == 'constant_with_warmup':
        return lambda step: min(1.0, float(step) / float(max(1, warmup_steps)))
    else:
        raise ValueError(f'Unknown scheduler type {scheduler_type}')


def get_tokenizer(config):
    tokenizer = T5Tokenizer.from_pretrained(config["plm_tokenizer_name"],
                                            model_max_length=config["model_max_length"])
    tokenizer.add_tokens(config["mention_start_token"])
    tokenizer.add_tokens(config["mention_end_token"])

    return tokenizer


class ASPT5Model(pl.LightningModule, gen.ASPGenerationMixin):
    main_input_name = "input_ids"

    def __init__(self,
                 config,
                 tokenizer: Optional[T5Tokenizer] = None) -> None:
        super().__init__()

        # Tokenizer setup
        if tokenizer is None:
            self.tokenizer = get_tokenizer(config)
        else:
            self.tokenizer = tokenizer
        self.mention_start_id = self.tokenizer.convert_tokens_to_ids(
            config["mention_start_token"])
        self.mention_end_id = self.tokenizer.convert_tokens_to_ids(
            config["mention_end_token"])

        # T5 Model setup
        self.t5 = T5Model.from_pretrained(
            config["plm_pretrained_name_or_path"])

        self.t5.resize_token_embeddings(self.tokenizer.vocab_size + 2)

        # Generation setup
        self.generation_config = GenerationConfig.from_model_config(
            self.t5.config)
        self.generation_config.early_stopping = True
        self.generation_config.max_new_tokens = 2048
        self.generation_config.num_beams = config["beam_size"]
        self.generation_config.num_return_sequences = config["beam_size"]
        self.generation_config.no_repeat_ngram_size = 0
        self.generation_config.encoder_no_repeat_ngram_size = 0
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_hidden_states = True
        self.generation_config.output_scores = True
        config["is_encoder_decoder"] = True
        config["vocab_size"] = self.tokenizer.vocab_size + 2

        # ASP settings
        self.num_labels = config["num_labels"]
        self.max_nested_depth = config["max_nest_depth"]

        # ASP modeling head
        dropout = nn.Dropout(config["asp_dropout_rate"])
        asp_hidden_dim = config["asp_hidden_dim"]
        asp_init_std = config["asp_init_std"]
        asp_activation = config["asp_activation"]

        self.action_head = utils.make_ffnn(feat_size=self.t5.config.d_model,
                                           hidden_size=[asp_hidden_dim],
                                           output_size=1,
                                           dropout=dropout,
                                           std=asp_init_std,
                                           activation=asp_activation)

        # left-right bracket
        self.lr_scorer = utils.make_ffnn(feat_size=2 * self.t5.config.d_model,
                                         hidden_size=[asp_hidden_dim],
                                         output_size=self.num_labels,
                                         dropout=dropout,
                                         std=asp_init_std,
                                         activation=asp_activation)

        self.save_hyperparameters(config)
        self.config = PretrainedConfig.from_dict(config)

        # F1 metrics
        self.val_f1 = F1ASP()
        self.test_f1 = F1ASP()

    def get_encoder(self):
        return self.t5.get_encoder()

    def get_decoder(self):
        return self.t5.get_decoder()

    @property
    def encoder(self):
        return self.get_encoder()

    @property
    def decoder(self):
        return self.get_decoder()

    def __get_params(self, named=False):
        plm_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if 't5' in name:
                to_add = (name, param) if named else param
                plm_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return plm_based_param, task_param

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        plm_param, task_param = self.__get_params(named=True)

        grouped_param = [{
            'params':
            [p for n, p in plm_param if not any(nd in n for nd in no_decay)],
            'lr':
            self.config.plm_learning_rate,
            'weight_decay':
            self.config.adam_weight_decay
        }, {
            'params':
            [p for n, p in plm_param if any(nd in n for nd in no_decay)],
            'lr':
            self.config.plm_learning_rate,
            'weight_decay':
            0.0
        }, {
            'params': [p for n, p in task_param],
            'lr': self.config.task_learning_rate,
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(grouped_param,
                                      lr=self.config.plm_learning_rate,
                                      eps=self.config.adam_eps,
                                      fused=self.config.fused)

        # Only warm up plm lr
        total_training_steps = self.config.train_len * self.config.num_epochs // (
            self.config.gradient_accumulation_steps * self.config.batch_size)
        warmup_steps = int(total_training_steps * self.config.warmup_ratio)

        lr_lambda_plm = get_scheduler_lambda(self.config.plm_scheduler,
                                             warmup_steps,
                                             total_training_steps)
        lr_lambda_task = get_scheduler_lambda(self.config.task_scheduler, 0,
                                              total_training_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [
                lr_lambda_plm,  # parameters with decay
                lr_lambda_plm,  # parameters without decay
                lr_lambda_task
            ])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self,
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                full_decoder_input_ids: Optional[torch.FloatTensor] = None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                cross_attn_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                encoder_input_ids=None,
                full_hidden_states: Optional[List[torch.FloatTensor]] = None,
                decoder_pairing=None,
                decoder_typing=None):
        return_dict = self.t5.config.use_return_dict
        outputs = self.t5.forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        assert isinstance(outputs, Seq2SeqModelOutput)

        outputs.last_hidden_state = outputs.last_hidden_state.to(self.device)
        batch_size = full_decoder_input_ids.size(0)
        output_ids = full_decoder_input_ids[:, 1:]  # excluding decoder BOS

        range_vec = torch.arange(output_ids.size(1) - 1,
                                 -1,
                                 -1,
                                 dtype=torch.long,
                                 device=self.device).unsqueeze(0).expand(
                                     batch_size, -1)

        if len(full_hidden_states) == 0:
            # the first valid token in the output
            denom = outputs.last_hidden_state.new_full(
                (batch_size, 1, self.num_labels), float("-inf"))
            l_choice = full_decoder_input_ids.new_full((batch_size, 1), -1)
            typing_choice = full_decoder_input_ids.new_full((batch_size, 1),
                                                            -1)
        else:
            # concatenating into sequence: (batch_size, seq_len, dim)
            decoder_output = torch.cat(full_hidden_states, dim=1)

            # check special tokens
            # Shape: (batch_size, seq_len, )
            is_l = (output_ids == self.mention_start_id)

            if is_l.sum() == 0:
                # no full mention and no previous mentions
                lr_denom = outputs.last_hidden_state.new_full(
                    (batch_size, 1, self.num_labels), float("-inf"))
                l_choice = full_decoder_input_ids.new_full((batch_size, 1), -1)
                typing_choice = full_decoder_input_ids.new_full(
                    (batch_size, 1), -1)
                denom = lr_denom
            else:
                # (batch_size, num_l, dim), (batch_size, num_l, )
                l_emb, _ = utils.batched_masked_select(decoder_output, is_l)

                # (batch_size, 1, num_l, )
                distance_to_previous_l, _ = utils.batched_masked_select(
                    range_vec, is_l)
                distance_to_previous_l = distance_to_previous_l.unsqueeze(1)

                # (batch_size, 1, num_l, 2*dim+feature_dim)
                lr_pair_emb = utils.prepare_pair_embeddings(
                    l_emb, outputs.last_hidden_state)

                # (batch_size, 1, num_l, self.num_labels)
                lr_score = self.lr_scorer(lr_pair_emb)

                num_l_each_instance = is_l.sum(dim=-1)

                for i in range(batch_size):
                    lr_score[i, :, :num_l_each_instance[i] -
                             self.max_nested_depth, :] = NEGINF
                    lr_score[i, :, num_l_each_instance[i]:, :] = NEGINF

                # (batch_size, 1)
                lr_denom = utils.logsumexp(lr_score, dim=(2, 3)).unsqueeze(-1)

                # (batch_size, 1, self.num_labels)
                lr_score_max_over_entities, max_l = lr_score.max(dim=2)
                # (batch_size, 1)
                typing_choice = lr_score_max_over_entities.argmax(dim=2)
                # (batch_size, 1)
                l_choice = max_l.squeeze(1).gather(1, typing_choice)
                denom = lr_denom

        action_logits = self.action_head(outputs.last_hidden_state)
        # (batch_size, 1, 1)
        action_logits = torch.cat(
            [torch.zeros_like(action_logits), action_logits, denom], dim=-1)
        # Restore lm_logits from action_logits
        # counting how many words have been copied
        is_copied = ((full_decoder_input_ids != self.mention_start_id) &\
                     (full_decoder_input_ids != self.mention_end_id))
        # (batch_size, 1)
        num_copied = is_copied.sum(dim=-1, keepdim=True) - 1
        num_copied = num_copied.clamp(max=encoder_input_ids.size(1) - 1)

        # compute pointer to input tokens
        lm_logits = action_logits.new_full(
            (action_logits.size(0), action_logits.size(1),
             self.config.vocab_size), float("-inf"))
        word_to_copy = encoder_input_ids.expand(num_copied.size(0),
                                                -1)  # repeating over beams
        word_to_copy = word_to_copy.gather(1, num_copied)

        lm_logits.scatter_(2, word_to_copy.unsqueeze(-1),
                           action_logits[:, :, :1])
        lm_logits[:, :, self.mention_start_id] = action_logits[:, :, 1]
        lm_logits[:, :, self.mention_end_id] = action_logits[:, :, 2]

        return ASPSeq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            pairing=l_choice,
            typing=typing_choice)

    def training_step(self, samples, batch_idx):
        """
        Training method
        """
        # process batch
        (doc_keys, subtoken_maps, batch) = samples
        input_ids = batch["input_ids"]
        attention_mask = batch["input_mask"]
        decoder_input_ids = batch["target_ids"]
        decoder_attention_mask = batch["target_mask"]
        action_labels = batch["action_labels"]
        lr_pair_flag = batch[
            "lr_pair_flag"]  # one-hot encoded labels for each lr pair

        flag_grad_ckpt = False
        if decoder_input_ids.size(1) > 2048:
            self.t5.gradient_checkpointing_enable()
            flag_grad_ckpt = True

        return_dict = self.t5.config.use_return_dict
        # decoder_input_ids starts with <pad> and has the same length as target_ids
        outputs = self.t5.forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=self.t5._shift_right(decoder_input_ids),
            decoder_attention_mask=decoder_attention_mask,
            use_cache=(not flag_grad_ckpt),
            return_dict=return_dict,
        )

        assert isinstance(outputs, Seq2SeqModelOutput)

        outputs.last_hidden_state = outputs.last_hidden_state.to(self.device)

        # shape: (batchsize, seq_len, 1)
        action_logits = self.action_head(outputs.last_hidden_state)

        # a new loss that is compatible with inference
        # batch_size, seq_len = decoder_output.size(0), decoder_output.size(1)
        target_mask = (decoder_input_ids != self.t5.config.pad_token_id)
        # (batch_size, seq_len)
        linearized_indices = (action_labels >= 0).cumsum(dim=-1) - 1
        # (batch_size, seq_len)
        is_l = (decoder_input_ids == self.mention_start_id)
        is_r = (decoder_input_ids == self.mention_end_id)

        if is_r.sum() == 0:
            numer, denom = (torch.full_like(outputs.last_hidden_state[..., :1],
                                            NEGINF),
                            torch.full_like(outputs.last_hidden_state[..., :1],
                                            NEGINF))
        else:
            # (batch_size, num_r / num_l)
            (l_pos,
             l_pos_mask) = utils.batched_masked_select(linearized_indices,
                                                       is_l)

            # (batch_size, num_r / num_l, hidden_dim)
            l_emb, _ = utils.batched_masked_select(outputs.last_hidden_state,
                                                   is_l)

            # (batch_size, seq_len, num_r / num_l)
            distance_to_previous_l = linearized_indices.unsqueeze(
                2) - l_pos.unsqueeze(1)

            # (batch_size, seq_len, num_r / num_l)
            is_after_l = (distance_to_previous_l > 0)
            is_after_l = is_after_l & target_mask.unsqueeze(
                2) & l_pos_mask.unsqueeze(1)

            # TODO: exchange for nucleus sampling?
            # check correctness for batch
            kept_l = min(self.max_nested_depth, l_emb.size(1))
            # (batch_size, seq_len, kept_l)
            _, prev_l_indices = (-distance_to_previous_l +
                                 (is_after_l * 10000)).topk(kept_l, dim=2)

            # (batch_size, seq_len, kept_l, hidden_dim)
            kept_l_emb = utils.batch_select(l_emb, prev_l_indices)
            # (batch_size, seq_len, kept_l)
            distance_to_previous_l = utils.dim_batched_index_select(
                distance_to_previous_l, prev_l_indices, dim=2)

            expanded_decoder_output = outputs.last_hidden_state.unsqueeze(
                2).expand(-1, -1, kept_l, -1)
            # shape(batch_size, seq_len, kept_l, 2*hidden_dim)
            lr_pair_emb = torch.cat([kept_l_emb, expanded_decoder_output],
                                    dim=-1)

            # shape(batch_size, seq_len, kept_l, num_labels)
            kept_is_after_l = is_after_l.gather(dim=2, index=prev_l_indices)
            lr_score = self.lr_scorer(
                lr_pair_emb) + (~kept_is_after_l).unsqueeze(-1) * NEGINF

            # (batch_size, seq_len, 1)
            lr_denom = utils.logsumexp(lr_score, dim=(
                2, 3), keepdim=False).unsqueeze(-1) * is_after_l.any(
                    dim=2, keepdim=True)
            # (batch_size, seq_len, num_l, num_labels) ->
            #     (batch_size, seq_len, kept_l, num_labels)
            kept_lr_pair_flag = utils.dim_batched_index_select(lr_pair_flag,
                                                               prev_l_indices,
                                                               dim=2)
            # (batch_size, seq_len, 1)
            lr_numer = utils.logsumexp(
                lr_score + (~kept_lr_pair_flag) * NEGINF,
                dim=(2, 3),
                keepdim=False).unsqueeze(-1) * is_after_l.any(dim=2,
                                                              keepdim=True)

            numer, denom = lr_numer, lr_denom

        # keeping <copy> score 0.
        action_logits = torch.cat(
            [torch.zeros_like(action_logits), action_logits], dim=-1)
        # Note: We want to compute a joint log-likelihood for action, boundary, and antecedent,
        # And we will use this joint LL for inference with beam search
        denom = utils.logsumexp(torch.cat([action_logits, denom], dim=-1),
                                dim=-1)
        numer = utils.logsumexp(torch.cat([
            action_logits + torch.where(
                utils.one_hot_ignore_negative(action_labels, num_classes=3),
                0., float("-inf"))[..., :2], numer
        ],
                                          dim=-1),
                                dim=-1)

        loss = (denom - numer)[decoder_attention_mask.bool()].sum()
        loss = loss / decoder_input_ids.size(0)

        if flag_grad_ckpt:
            self.t5.gradient_checkpointing_disable()
            flag_grad_ckpt = False

        self.log("train_loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        return {"loss": loss}

    def get_mapping_to_input_sequence(self, output_ids):
        # Get the mapping from the output with special tokens
        # to the input without special tokens.
        mapping, new_id = [], -1
        for i in range(len(output_ids)):
            if output_ids[i] == self.mention_start_id:
                new_id += 1
            elif output_ids[i] == self.mention_end_id:
                new_id += 0
            else:
                new_id += 1
            mapping.append(new_id)
            if output_ids[i] == self.mention_start_id:
                new_id -= 1

        return mapping

    def validation_step(self, samples, batch_idx):
        """
        Validation step
        """
        # process batch
        (doc_keys, subtoken_maps, batch) = samples
        input_ids = batch["input_ids"]
        to_copy_ids = batch["to_copy_ids"]
        target_ids = batch["target_ids"]
        ent_types = batch["ent_types"]

        # save the decoded actions
        decoder_pairing, decoder_typing = [], []
        model_output = self.generate(input_ids,
                                     generation_config=self.generation_config,
                                     **{
                                         "decoder_encoder_input_ids":
                                         to_copy_ids,
                                         "decoder_pairing": decoder_pairing,
                                         "decoder_typing": decoder_typing
                                     })
        # taking the best sequence in the beam, removing </s>
        for idx in range(input_ids.size(0)):
            subtoken_map = subtoken_maps[idx]
            idx_target_ids = target_ids[idx]
            labels = ent_types[idx]
            # targets
            mapping = self.get_mapping_to_input_sequence(idx_target_ids)
            targets, start_ind = [], []

            # TODO: allow nested mentions
            # reconstructing mention_indices and antecedent_indices
            for i in range(len(idx_target_ids)):
                if idx_target_ids[i] == self.mention_start_id:
                    start_ind.append(i)
                if idx_target_ids[i] == self.mention_end_id:
                    entity = (
                        int(subtoken_map[int(
                            mapping[start_ind[-1]])]),  # no nested
                        int(subtoken_map[int(mapping[i])]),
                        int(labels[i]))
                    targets.append(entity)

            # TODO: allow nested mentions
            # predictions
            output_ids = model_output.sequences[idx][1:]
            pairing = [x[idx] for x in decoder_pairing]
            typing = [x[idx] for x in decoder_typing]
            mapping = self.get_mapping_to_input_sequence(output_ids)
            preds, start_ind = [], []
            # reconstructing mention_indices and antecedent_indices
            for i in range(len(output_ids)):
                if output_ids[i] == self.tokenizer.pad_token_id:
                    break
                if output_ids[i] == self.mention_start_id:
                    start_ind.append(i)
                if output_ids[i] == self.mention_end_id:
                    this_type = int(typing[i])
                    entity = (subtoken_map[mapping[start_ind[pairing[i]]]],
                              subtoken_map[mapping[i]], this_type)
                    preds.append(entity)
            self.val_f1.update(preds, targets)
        self.log("val_f1",
                 self.val_f1,
                 prog_bar=True,
                 logger=True,
                 on_epoch=True,
                 on_step=True)

    def test_step(self, samples, batch_idx):
        """
        Test step
        """
        # process batch
        (doc_keys, subtoken_maps, batch) = samples
        input_ids = batch["input_ids"]
        to_copy_ids = batch["to_copy_ids"]
        target_ids = batch["target_ids"]
        ent_types = batch["ent_types"]

        # save the decoded actions
        decoder_pairing, decoder_typing = [], []
        model_output = self.generate(input_ids,
                                     generation_config=self.generation_config,
                                     **{
                                         "decoder_encoder_input_ids":
                                         to_copy_ids,
                                         "decoder_pairing": decoder_pairing,
                                         "decoder_typing": decoder_typing
                                     })
        # taking the best sequence in the beam, removing </s>
        for idx in range(input_ids.size(0)):
            subtoken_map = subtoken_maps[idx]
            idx_target_ids = target_ids[idx]
            labels = ent_types[idx]
            # targets
            mapping = self.get_mapping_to_input_sequence(idx_target_ids)
            targets, start_ind = [], []

            # TODO: allow nested mentions
            # reconstructing mention_indices and antecedent_indices
            for i in range(len(idx_target_ids)):
                if idx_target_ids[i] == self.mention_start_id:
                    start_ind.append(i)
                if idx_target_ids[i] == self.mention_end_id:
                    entity = (
                        int(subtoken_map[int(
                            mapping[start_ind[-1]])]),  # no nested
                        int(subtoken_map[int(mapping[i])]),
                        int(labels[i]))
                    targets.append(entity)

            # TODO: allow nested mentions
            # predictions
            output_ids = model_output.sequences[idx][1:]
            pairing = [x[idx] for x in decoder_pairing]
            typing = [x[idx] for x in decoder_typing]
            mapping = self.get_mapping_to_input_sequence(output_ids)
            preds, start_ind = [], []
            # reconstructing mention_indices and antecedent_indices
            for i in range(len(output_ids)):
                if output_ids[i] == self.tokenizer.pad_token_id:
                    break
                if output_ids[i] == self.mention_start_id:
                    start_ind.append(i)
                if output_ids[i] == self.mention_end_id:
                    this_type = int(typing[i])
                    entity = (subtoken_map[mapping[start_ind[pairing[i]]]],
                              subtoken_map[mapping[i]], this_type)
                    preds.append(entity)
            self.test_f1.update(preds, targets)
        self.log("test_f1",
                 self.test_f1,
                 prog_bar=True,
                 logger=True,
                 on_epoch=True,
                 on_step=True)