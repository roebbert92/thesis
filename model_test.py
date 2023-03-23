from models.t5_model import ASP_T5

model = ASP_T5.from_pretrained("t5-small", num_labels=6, device="cpu")