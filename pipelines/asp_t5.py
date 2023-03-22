import lightning.pytorch as pl
from transformers import T5Tokenizer
import torch

from models.t5_model import ASP_T5


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


class ASP_T5_Pipeline(pl.LightningModule):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config["plm_tokenizer_name"])
        self.tokenizer.add_tokens(self.config["mention_start_token"])
        self.tokenizer.add_tokens(self.config["mention_end_token"])

        self.mention_start_id = self.tokenizer.convert_tokens_to_ids(
            self.config["mention_start_token"])
        self.mention_end_id = self.tokenizer.convert_tokens_to_ids(
            self.config["mention_end_token"])

        self.model = ASP_T5.from_pretrained(
            config['plm_pretrained_name_or_path'],
            asp_hidden_dim=config["asp_hidden_size"],
            asp_dropout_rate=config["asp_dropout_rate"],
            asp_init_std=config["asp_init_std"],
            asp_activation=config["asp_activation"],
            num_labels=config["num_labels"],
            mention_start_id=self.mention_start_id,
            mention_end_id=self.mention_end_id,
            max_nest_depth=config["max_nest_depth"])

        self.beam_size = config["beam_size"]
        self.model.resize_token_embeddings(self.tokenizer.vocab_size + 2)

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        plm_param, task_param = self.model.get_params(named=True)

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
                                      lr=self.config['plm_learning_rate'],
                                      eps=self.config["adam_eps"])

        # Only warm up plm lr
        total_training_steps = self.config["train_len"] * self.config[
            "num_epochs"] // (  #self.config["gradient_accumulation_steps"] *
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

    def training_step(self, batch, batch_idx):
        """
        Training method
        """
        input_ids = batch["input_ids"]
        input_mask = batch["input_mask"]
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]
        action_labels = batch["action_labels"]
        lr_pair_flag = batch["lr_pair_flag"]

        flag_grad_ckpt = False
        if target_ids.size(1) > 2048:
            self.model.gradient_checkpointing_enable()
            flag_grad_ckpt = True

        seq2seq_output = self.model.forward(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            decoder_input_ids=target_ids,
                                            decoder_attention_mask=target_mask,
                                            labels=action_labels,
                                            output_hidden_states=True,
                                            lr_pair_flag=lr_pair_flag,
                                            use_cache=(not flag_grad_ckpt))
        if flag_grad_ckpt:
            self.model.gradient_checkpointing_disable()
            flag_grad_ckpt = False
        total_loss = seq2seq_output.loss

        return total_loss