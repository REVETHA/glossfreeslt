import argparse
from pathlib import Path
import sys

import torch
import yaml
from transformers import MBartTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataloader.datasets import S2T_Dataset
from definition import PAD_IDX
from models.models import gloss_free_model
import utils


ROOT = Path(__file__).resolve().parents[1]
with (ROOT / "configs" / "isign" / "config1.yaml").open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config = utils.expand_env_vars(config)

args = argparse.Namespace(
    input_size=224,
    resize=256,
    model_type="gfslt",
    frozenFeatureExtractor=False,
)
tokenizer = MBartTokenizer.from_pretrained(
    config["model"]["tokenizer"],
    src_lang=config["data"]["language"],
    tgt_lang=config["data"]["language"],
)
tokenizer.model_max_length = config["data"]["max_length"]

dataset = S2T_Dataset(
    path=config["data"]["train_label_path"],
    tokenizer=tokenizer,
    config=config,
    args=args,
    phase="train",
)
print(f"dataset_length: {len(dataset)}", flush=True)
sample = dataset[0]
print(f"sample_name: {sample[0]}", flush=True)
print(f"sample_frames_shape: {tuple(sample[1].shape)}", flush=True)
print(f"sample_text: {sample[2]}", flush=True)
batch = dataset.collate_fn([sample])
src_input, tgt_input = batch
print(f"src_input_ids_shape: {tuple(src_input['input_ids'].shape)}", flush=True)
print(f"src_length_batch: {src_input['src_length_batch'].tolist()}", flush=True)
print(f"src_attention_mask_shape: {tuple(src_input['attention_mask'].shape)}", flush=True)
print(f"tgt_input_ids_shape: {tuple(tgt_input['input_ids'].shape)}", flush=True)

model = gloss_free_model(config, args).cuda()
model.train()
print("forward_start", flush=True)
try:
    out_logits, frame_features = model(src_input, tgt_input)
except Exception:
    print("FORWARD_FAILED", flush=True)
    raise
print("FORWARD_SUCCESS", flush=True)
print(f"model_output_logits_shape: {tuple(out_logits.shape)}", flush=True)
print(f"frame_features_shape: {tuple(frame_features.shape)}", flush=True)
print(f"allocated_memory_mb_before_backward: {torch.cuda.memory_allocated() / 1024**2:.2f}", flush=True)
print(f"reserved_memory_mb_before_backward: {torch.cuda.memory_reserved() / 1024**2:.2f}", flush=True)

criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
labels = tgt_input["input_ids"].reshape(-1).cuda(non_blocking=True)
logits = out_logits.reshape(-1, out_logits.shape[-1])
loss = criterion(logits, labels)
print(f"loss: {loss.item():.6f}", flush=True)
loss.backward()

nonzero_gradients = sum(
    1 for parameter in model.parameters()
    if parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
)
print("BACKWARD_SUCCESS", flush=True)
print(f"parameters_with_nonzero_gradients: {nonzero_gradients}", flush=True)
print(f"allocated_memory_mb_after_backward: {torch.cuda.memory_allocated() / 1024**2:.2f}", flush=True)
print(f"reserved_memory_mb_after_backward: {torch.cuda.memory_reserved() / 1024**2:.2f}", flush=True)
