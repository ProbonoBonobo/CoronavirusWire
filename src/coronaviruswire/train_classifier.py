from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import torch
from fast_bert.data_cls import BertDataBunch
import os

base_path = os.path.abspath("../../lib/fast-bert")
DATA_PATH = os.path.join(base_path, "data")
LABEL_PATH = os.path.join(base_path, "labels")
OUTPUT_DIR = os.path.abspath(".")
logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{"name": "accuracy", "function": accuracy}]

databunch = BertDataBunch(
    DATA_PATH,
    LABEL_PATH,
    tokenizer="bert-base-uncased",
    train_file="train.csv",
    val_file="val.csv",
    label_file="labels.csv",
    text_col="text",
    label_col=[
        "domestic",
        "county",
        "city",
        "regional",
        "state",
        "national",
        "international",
        "is_relevant",
    ],
    batch_size_per_gpu=1,
    max_seq_length=2,
    multi_gpu=False,
    multi_label=True,
    model_type="bert",
)

learner = BertLearner.from_pretrained_model(
    databunch,
    pretrained_path="bert-base-uncased",
    metrics=metrics,
    device=device_cuda,
    logger=logger,
    output_dir=OUTPUT_DIR,
    finetuned_wgts_path=None,
    multi_gpu=False,
    is_fp16=False,
    multi_label=True,
    logging_steps=50,
)

learner.fit(
    epochs=6,
    lr=6e-5,
    # validate=True, 	# Evaluate the model after each epoch
    # # schedule_type="warmup_cosine",
    optimizer_type="adamW",
)
learner.save_model()
