"""Export the CLIP (clip-ViT-B-32) TEXT encoder to a small ONNX model so the web app does not need PyTorch at run time
(PyTorch + CLIP needs ~2 GB RAM; this needs ~150 MB, so the app fits Render's free 512 MB instance).

Pipeline: PyTorch -> fp32 ONNX -> weight-only 8-bit MatMulNBits (activations stay float32; plain dynamic int8 destroyed accuracy: cosine 0.88)
          -> token-embedding table to int8 per row.  Writes models/clip_text.onnx (<100 MB, GitHub's file limit) and models/tokenizer.json,
          then checks parity against the original encoder and refuses to finish if it is not close.
Run from moments/ (needs torch, sentence-transformers, onnx, onnx-ir, onnxruntime, tokenizers):  python scripts/export_text_encoder.py
"""
import sys
from pathlib import Path
import numpy as np, torch, onnx
from onnx import helper, numpy_helper, TensorProto

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "models"; OUT.mkdir(exist_ok=True)
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("clip-ViT-B-32", device="cpu")
hf = st[0].model.eval()                              # transformers CLIPModel
tok = st[0].processor.tokenizer


class TextTower(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, input_ids, attention_mask):
        out = self.m.text_model(input_ids=input_ids, attention_mask=attention_mask)
        return self.m.text_projection(out.pooler_output)        # pooled end-of-text token -> 512-d joint space


ids = tok(["a photo of a beach"], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
fp32 = OUT / "_fp32.onnx"; nbits = OUT / "_nbits8.onnx"; final = OUT / "clip_text.onnx"
torch.onnx.export(TextTower(hf), (ids["input_ids"], ids["attention_mask"]), str(fp32), input_names=["input_ids", "attention_mask"],
                  output_names=["text_embeds"], dynamic_axes={"input_ids": {0: "b"}, "attention_mask": {0: "b"}, "text_embeds": {0: "b"}},
                  opset_version=17, dynamo=False)
tok.backend_tokenizer.save(str(OUT / "tokenizer.json"))

# ---- weight-only 8-bit for every MatMul
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=8, block_size=32, is_symmetric=True, accuracy_level=4)
q.process(); q.model.save_model_to_file(str(nbits), use_external_data_format=False)

# ---- token-embedding table (49408 x 512 float32 = 100 MB) -> int8 with one scale per row
m = onnx.load(str(nbits)); g = m.graph; inits = {i.name: i for i in g.initializer}; new_nodes, done = [], False
for n in g.node:
    if (not done) and n.op_type == "Gather" and n.input[0] in inits and list(inits[n.input[0]].dims) == [49408, 512]:
        W = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
        scale = (np.abs(W).max(axis=1, keepdims=True) / 127.0).clip(min=1e-8).astype(np.float32)
        Wq = np.round(W / scale).astype(np.int8)
        qn, sn = n.input[0] + "_q8", n.input[0] + "_scale"
        g.initializer.extend([numpy_helper.from_array(Wq, qn), numpy_helper.from_array(scale, sn)])
        g.initializer.remove(inits[n.input[0]])
        a, b, c = n.output[0] + "_g8", n.output[0] + "_gs", n.output[0] + "_gf"
        new_nodes += [helper.make_node("Gather", [qn, n.input[1]], [a], axis=0), helper.make_node("Gather", [sn, n.input[1]], [b], axis=0),
                      helper.make_node("Cast", [a], [c], to=TensorProto.FLOAT), helper.make_node("Mul", [c, b], [n.output[0]])]
        done = True
    else: new_nodes.append(n)
if not done: sys.exit("FATAL: token-embedding Gather not found")
del g.node[:]; g.node.extend(new_nodes)
onnx.save(m, str(final))
for f in (fp32, nbits): f.unlink(missing_ok=True)
for f in OUT.glob("_*"): f.unlink()
for f in OUT.glob("clip_text_int8*"): f.unlink()
print(f"wrote {final.name} ({final.stat().st_size/1e6:.0f} MB) and tokenizer.json")
if final.stat().st_size > 95e6: sys.exit("FATAL: model is too large for a plain GitHub file (limit 100 MB)")

# ---- parity check against the original encoder
sys.path.insert(0, str(ROOT / "app"))
import textenc
prompts = ["goa", "bangalore", "a beach with sea and sand", "wide sandy beach with palm trees and blue ocean waves under bright sun",
           "snowy mountains, ice, cold grey northern coastline, rocky cliffs, dark forest", "a photo of a paper document, a receipt, a form or a screenshot of text",
           "colorful colonial style buildings with ornate balconies and narrow cobblestone streets", "people at a party, wedding, concert or birthday celebration",
           "job i need to apply", "a pet dog or cat", "close up of spicy fish curry served on a plate with steamed rice", "x"]
ref = st.encode(prompts, convert_to_numpy=True, normalize_embeddings=True)
new = np.stack([textenc.encode(p) for p in prompts])
cos = (ref * new).sum(1)
print("cosine(original, onnx) per prompt: min %.4f mean %.4f" % (cos.min(), cos.mean()))
E = np.load(ROOT / "data" / "embeddings.npy")
ov = [len(set(np.argsort(-(E @ r))[:50]) & set(np.argsort(-(E @ n))[:50])) / 50 for r, n in zip(ref, new)]
print("top-50 photo overlap with the original encoder: min %.2f mean %.2f" % (min(ov), float(np.mean(ov))))
if cos.min() < 0.99 or float(np.mean(ov)) < 0.95 or min(ov) < 0.8: sys.exit("FATAL: the compressed encoder is not close enough to the original; do not ship it")
print("OK parity")
