import argparse
from pathlib import Path
import sys

import torch
import yaml
from transformers import MBartTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.models import gloss_free_model
import utils


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "isign" / "config1.yaml"

with CONFIG_PATH.open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config = utils.expand_env_vars(config)

print(f"loading tokenizer: {config['model']['tokenizer']}", flush=True)
tokenizer = MBartTokenizer.from_pretrained(
    config["model"]["tokenizer"],
    src_lang=config["data"]["language"],
    tgt_lang=config["data"]["language"],
)
print(f"tokenizer_load: SUCCESS; vocab_size={len(tokenizer)}", flush=True)

args = argparse.Namespace(model_type="gfslt", frozenFeatureExtractor=False)
print("constructing GFSLT model on CUDA", flush=True)
model = gloss_free_model(config, args)
model.cuda()

print("MODEL_CONSTRUCTION_SUCCESS", flush=True)
print(model)
print(f"allocated_memory_mb: {torch.cuda.memory_allocated() / 1024**2:.2f}")
print(f"reserved_memory_mb: {torch.cuda.memory_reserved() / 1024**2:.2f}")
print(f"total_gpu_memory_mb: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f}")
