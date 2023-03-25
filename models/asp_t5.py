from typing import Any
import lightning.pytorch as pl
from transformers import T5Tokenizer, T5Model
import torch
import torch.nn as nn

import models.utils as utils

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
                                            model_max_length=4096)
    tokenizer.add_tokens(config["mention_start_token"])
    tokenizer.add_tokens(config["mention_end_token"])

    return tokenizer


class ASPT5Model(pl.LightningModule):
    def __init__(self, config, tokenizer) -> None:
        super().__init__()
        self.config = config

        # Tokenizer setup
        self.tokenizer = tokenizer
        self.mention_start_id = tokenizer.convert_tokens_to_ids(
            self.config["mention_start_token"])
        self.mention_end_id = tokenizer.convert_tokens_to_ids(
            self.config["mention_end_token"])

        # T5 Model setup
        self.t5 = T5Model.from_pretrained(
            self.config["plm_pretrained_name_or_path"])

        self.t5.resize_token_embeddings(self.tokenizer.vocab_size + 2)

        # ASP settings
        self.num_labels = self.config["num_labels"]
        self.max_nested_depth = self.config["max_nest_depth"]

        # ASP modeling head
        dropout = nn.Dropout(self.config["asp_dropout_rate"])
        asp_hidden_dim = self.config["asp_hidden_dim"]
        asp_init_std = self.config["asp_init_std"]
        asp_activation = self.config["asp_activation"]

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
            self.config['plm_learning_rate'],
            'weight_decay':
            self.config['adam_weight_decay']
        }, {
            'params':
            [p for n, p in plm_param if any(nd in n for nd in no_decay)],
            'lr':
            self.config['plm_learning_rate'],
            'weight_decay':
            0.0
        }, {
            'params': [p for n, p in task_param],
            'lr': self.config['task_learning_rate'],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(grouped_param,
                                      lr=self.config["plm_learning_rate"],
                                      eps=self.config["adam_eps"],
                                      fused=self.config["fused"])

        # Only warm up plm lr
        total_training_steps = self.config["train_len"] * self.config[
            "num_epochs"] // (self.config["gradient_accumulation_steps"] *
                              self.config["batch_size"])
        warmup_steps = int(total_training_steps * self.config['warmup_ratio'])

        lr_lambda_plm = get_scheduler_lambda(self.config['plm_scheduler'],
                                             warmup_steps,
                                             total_training_steps)
        lr_lambda_task = get_scheduler_lambda(self.config['task_scheduler'], 0,
                                              total_training_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            [
                lr_lambda_plm,  # parameters with decay
                lr_lambda_plm,  # parameters without decay
                lr_lambda_task
            ])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, samples, batch_idx):
        """
        Training method
        """
        # process batch
        (doc_keys, batch) = samples
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
        decoder_input_ids = self.t5._shift_right(decoder_input_ids)

        outputs = self.t5.forward(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=(not flag_grad_ckpt),
            return_dict=return_dict,
        )

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

            # shape(batch_size, seq_len, kept_l, num_typing_classes)
            kept_is_after_l = is_after_l.gather(dim=2, index=prev_l_indices)
            lr_score = self.lr_scorer(
                lr_pair_emb) + (~kept_is_after_l).unsqueeze(-1) * NEGINF

            # (batch_size, seq_len, 1)
            lr_denom = utils.logsumexp(lr_score, dim=(
                2, 3), keepdim=False).unsqueeze(-1) * is_after_l.any(
                    dim=2, keepdim=True)
            # (batch_size, seq_len, num_l, num_typing_classes) ->
            #     (batch_size, seq_len, kept_l, num_typing_classes)
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
        assert isinstance(loss, torch.Tensor)

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
