from pathlib import Path
import gzip
import json
import os
import pickle

from transformers import MBartConfig, MBartForConditionalGeneration, MBartTokenizer
from hftrim.ModelTrimmers import MBartTrimmer
from hftrim.TokenizerTrimmer import TokenizerTrimmer


ISIGN_LMDB_ROOT = os.environ.get("ISIGN_LMDB_ROOT", r"D:\iSign_LMDB")
MBART_MODELS_ROOT = os.environ.get("MBART_MODELS_ROOT", r"E:\FINAL YEAR PROJECT\GLOSS FREE SLT\mbart_models")

LABELS = Path(ISIGN_LMDB_ROOT) / "labels" / "labels.train"
BASE = Path(MBART_MODELS_ROOT) / "_official_mbart_large_cc25"
OUTPUT = Path(MBART_MODELS_ROOT)
MYTRAN_CONFIG = {
    "_name_or_path": "facebook/mbart-large-cc25",
    "_num_labels": 3,
    "activation_dropout": 0,
    "activation_function": "relu",
    "add_bias_logits": False,
    "add_final_layer_norm": True,
    "architectures": ["MBartForConditionalGeneration"],
    "attention_dropout": 0,
    "bos_token_id": 0,
    "d_model": 1024,
    "decoder_attention_heads": 8,
    "decoder_ffn_dim": 4096,
    "decoder_layerdrop": 0.0,
    "decoder_layers": 3,
    "dropout": 0.1,
    "encoder_attention_heads": 8,
    "encoder_ffn_dim": 4096,
    "encoder_layerdrop": 0.0,
    "encoder_layers": 3,
    "eos_token_id": 2,
    "forced_eos_token_id": 2,
    "id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"},
    "init_std": 0.02,
    "is_encoder_decoder": True,
    "label2id": {"LABEL_0": 0, "LABEL_1": 1, "LABEL_2": 2},
    "max_length": 1024,
    "max_position_embeddings": 1024,
    "model_type": "mbart",
    "normalize_before": True,
    "normalize_embedding": True,
    "num_beams": 4,
    "num_hidden_layers": 3,
    "output_past": True,
    "pad_token_id": 1,
    "scale_embedding": True,
    "static_position_embeddings": False,
    "task_specific_params": {"translation_en_to_ro": {"decoder_start_token_id": 250020}},
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.22.2",
    "use_cache": True,
    "vocab_size": 2454,
}

os.makedirs("/tmp", exist_ok=True)

print(f"loading iSign labels: {LABELS}", flush=True)
with gzip.open(LABELS, "rb") as stream:
    raw_data = pickle.load(stream)
data = [value["text"] for value in raw_data.values()]
print(f"label records: {len(data)}; language: en_XX", flush=True)

print(f"loading official tokenizer: {BASE}", flush=True)
tokenizer = MBartTokenizer.from_pretrained(str(BASE), src_lang="en_XX", tgt_lang="en_XX")
trimmed_tokenizer_builder = TokenizerTrimmer(tokenizer)
trimmed_tokenizer_builder.make_vocab(data)
trimmed_tokenizer_builder.make_tokenizer()

print(f"loading official model: {BASE}", flush=True)
model = MBartForConditionalGeneration.from_pretrained(str(BASE))
trimmed_model_builder = MBartTrimmer(
    model,
    model.config,
    trimmed_tokenizer_builder.trimmed_tokenizer,
)
trimmed_model_builder.make_weights(trimmed_tokenizer_builder.trimmed_vocab_ids)
trimmed_model_builder.make_model()

trimmed_tokenizer = trimmed_tokenizer_builder.trimmed_tokenizer
trimmed_model = trimmed_model_builder.trimmed_model
mbart_output = OUTPUT / "MBart_trimmed"
mytran_output = OUTPUT / "mytran"
mbart_output.mkdir(parents=True, exist_ok=True)
mytran_output.mkdir(parents=True, exist_ok=True)

print(f"saving MBart_trimmed: {mbart_output}", flush=True)
trimmed_tokenizer.save_pretrained(str(mbart_output))
trimmed_model.save_pretrained(str(mbart_output))

print("loading official mytran architecture config", flush=True)
configuration = MBartConfig.from_dict(MYTRAN_CONFIG)
configuration.vocab_size = trimmed_model.config.vocab_size
mytran_model = MBartForConditionalGeneration._from_config(configuration)
mytran_model.model.shared = trimmed_model.model.shared

print(f"saving mytran: {mytran_output}", flush=True)
mytran_model.save_pretrained(str(mytran_output))
print("asset preparation complete", flush=True)
