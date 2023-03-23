import sys

sys.path.append("/Users/robinloebbert/Masterarbeit/thesis")
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import unittest
import pickle
from models.t5_model import ASP_T5
from data_preprocessing.tensorize import ner_collate_fn


class ASPT5ModelTest(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.model = ASP_T5.from_pretrained("t5-small", num_labels=6).to("cpu")

    def test_single_batch(self):
        with open("tests/data/wnut_batch_1.pkl", "rb") as file:
            data_point = pickle.load(file)
        batch = ner_collate_fn([data_point])[1]
        self.model.train()

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
        self.assertIsNotNone(total_loss)
        self.assertGreater(total_loss.item(), 0.0)


if __name__ == '__main__':
    unittest.main()