"""CLIP text encoder on ONNX Runtime (no PyTorch): ~150 MB RAM instead of ~2 GB. Produces unit-length 512-d vectors that live in the same
space as data/embeddings.npy (made by the original encoder). Export it with scripts/export_text_encoder.py."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
_MODEL = ROOT / "models" / "clip_text.onnx"
_TOK = ROOT / "models" / "tokenizer.json"
if not _MODEL.exists() or not _TOK.exists():
    raise SystemExit(f"FATAL: {_MODEL} / {_TOK} missing. Run: python scripts/export_text_encoder.py")

_tok = Tokenizer.from_file(str(_TOK))
_tok.enable_truncation(max_length=77)
_tok.enable_padding(length=77, pad_id=49407, pad_token="<|endoftext|>")      # CLIP pads with the end-of-text token
_opts = ort.SessionOptions(); _opts.intra_op_num_threads = 1; _opts.inter_op_num_threads = 1; _opts.enable_mem_pattern = False
_sess = ort.InferenceSession(str(_MODEL), _opts, providers=["CPUExecutionProvider"])


def encode(text):
    e = _tok.encode(str(text))
    ids = np.array([e.ids], dtype=np.int64); mask = np.array([e.attention_mask], dtype=np.int64)
    v = _sess.run(None, {"input_ids": ids, "attention_mask": mask})[0][0].astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)
