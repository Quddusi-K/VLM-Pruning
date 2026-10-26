import json

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# Phase A (v2) — per-image visual-token exit: oracle + signal profiling on POPE, GQA, TextVQA, MME

**Question:** is there a cheap, training-free signal that tells, *per image*, when all visual tokens can be dropped
("visual exit") — well enough to beat a single fixed exit layer for the whole dataset (VTW-style)?

**What one profiling pass computes per example** (single prefill, ~2 s on 2×T4 in fp16):
- **Oracle:** for every exit layer E = 0…31 (layers 0…E−1 see the image, layers E…31 don't), the model's first answer
  token. Done exactly and cheaply: the full-sequence pass is shared, and each "exit at E" branch only re-runs layers E…31
  on the ~25 text tokens. `preserved[E]` = same first token as the full model.
- **Candidate signals at every layer E** (all computable *before* layer E runs, using 1–30 query rows, no attention matrix):
  - `cf_*` — local counterfactual: change in the attention update when visual keys are removed (relative to the residual
    stream `_res` or to the attention output `_attn`), for the last token or averaged over the question tokens.
  - `mass_*` — share of attention going to visual tokens (last token / question tokens).
  - `lens_*` — logit lens on the last token: KL to the previous layer, top-1 probability.

**Outputs:** `/kaggle/working/phaseA/profile_{pope,gqa}.jsonl` (resumable), `verify_*.jsonl`, and `phaseA_results.zip`.
The analysis section (CPU only) runs the go/no-go test. **Send back the zip** (or the analysis output).

**Gate:** pass if some signal reaches the same held-out fidelity (agreement with the full model) as the best fixed exit
layer at clearly lower FLOPs on both POPE and GQA, and beats a shuffled-exit control.

Run order: §1–§7 (GPU), then §8 (CPU analysis; can be re-run alone later from the JSONL files).

**v2 changes:** TextVQA + MME data (§6b), a restore cell for the POPE/GQA profiles you already have (§6c),
generalized profiling/verification over all benchmarks, analysis loads every `profile_*.jsonl`, and a new §8.5
that tests the headline claim: **one global threshold for all tasks** vs one fixed exit layer for all tasks.
''')

md("## 1. Environment check")
code(r'''
import subprocess, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("CUDA:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
''')

md("## 2. Install FastV's transformers fork (+ multi-GPU fix). Restart the kernel afterwards if `transformers` was already imported.")
code(r'''
%%bash
cd /kaggle/working
if [ ! -d "FastV" ]; then
  git clone --depth 1 https://github.com/pkunlp-icler/FastV.git
fi
cd /kaggle/working/FastV/src/FastV/llava-hf/transformers/src/transformers/models/llama
git checkout modeling_llama.py
python3 -c '
from pathlib import Path
p = Path("modeling_llama.py")
code = p.read_text()
old = "keep_indexs = torch.cat( (torch.arange(FASTV_image_token_start_index,device=device), top_attention_rank_index, torch.arange(FASTV_image_token_start_index+FASTV_image_token_length,seq_length_with_past,device=device)))"
new = "keep_indexs = torch.cat( (torch.arange(FASTV_image_token_start_index,device=hidden_states.device), top_attention_rank_index.to(hidden_states.device), torch.arange(FASTV_image_token_start_index+FASTV_image_token_length,seq_length_with_past,device=hidden_states.device)))"
assert old in code, "target line not found"
p.write_text(code.replace(old, new))
print("multi-GPU fix applied")
'
cd /kaggle/working/FastV/src/FastV/llava-hf/transformers
pip install -e . --no-deps -q
pip install -q "tokenizers<0.19,>=0.14" pillow accelerate sentencepiece datasets
python -c "import transformers; print('transformers', transformers.__version__)"
''')

md("## 3. Imports, paths, constants")
code(r'''
import os
os.environ["USE_TF"] = "0"
import gc, json, math, random, re, string, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaProcessor, LlavaForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPast
import transformers.models.llama.modeling_llama as modeling_llama
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv

print("transformers", transformers.__version__)
assert hasattr(modeling_llama.LlamaModel, "fastv_forward"), "Not FastV's fork: run §2 and restart the kernel."

WORK = Path("/kaggle/working")
DATA, OUT = WORK / "data", WORK / "phaseA"
for p in (DATA, OUT):
    p.mkdir(parents=True, exist_ok=True)

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_START, IMAGE_LEN = 4, 576
N_LAYERS, D_MODEL, D_FFN = 32, 4096, 11008
''')

md(r'''
## 4. Load the model once — **fp16, no quantization**
fp16 is faster than 8-bit for the many short exit branches, and it is what Phase B will use. Baseline numbers can differ
slightly from the 8-bit Phase 1 runs; Phase A computes its own baseline, so this does not matter here.
''')
code(r'''
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
processor = LlavaProcessor(image_processor=CLIPImageProcessor.from_pretrained(MODEL_ID), tokenizer=tokenizer)

if "model" not in globals():
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="eager", device_map="auto")
    model.eval()
model.config.fastv_config = None
LM = model.language_model          # LlamaForCausalLM (has .model and .lm_head)
print("Devices:", sorted({str(p.device) for p in model.parameters()}))
''')

code(r'''
def set_method(cfg):
    model.config.fastv_config = cfg

def build_inputs(image, question, suffix=""):
    prompt = f"USER: <image>\n{question}{suffix}\nASSISTANT:"
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    pos = (inputs["input_ids"][0] == model.config.image_token_index).nonzero().flatten()
    assert pos.numel() == 1 and pos.item() == IMAGE_START, f"image token at {pos.tolist()}"
    return inputs

@torch.no_grad()
def generate(inputs, max_new_tokens):
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=False)
    return processor.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

def free_mem():
    gc.collect(); torch.cuda.empty_cache()

def layer_flops(n):
    return 4 * n * D_MODEL**2 + 2 * n * n * D_MODEL + 2 * n * D_MODEL * D_FFN

def exit_flops(n, E):
    """Relative prefill FLOPs when layers 0..E-1 see all n tokens and layers E..31 see only text."""
    return (E * layer_flops(n) + (N_LAYERS - E) * layer_flops(n - IMAGE_LEN)) / (N_LAYERS * layer_flops(n))

def parse_yes_no(text):
    t = text.strip().lower()
    if t.startswith("yes"): return "yes"
    if t.startswith("no"):  return "no"
    m = re.search(r"\b(yes|no)\b", t)
    return m.group(1) if m else "no"

_ARTICLES = {"a", "an", "the"}
def norm_answer(s):
    s = s.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in s.split() if w not in _ARTICLES)
''')

md(r'''
## 5. Profiling forward (exact oracle + signals in one prefill)
Installed through the same `fastv_forward` dispatcher idea as before (`method="profile"`), so LLaVA's own code merges the
image features and hands us the merged embeddings. FastV itself stays available as `method="fastv"` (used in §7).
''')
code(r'''
PROFILE = {}

def _dev(module):
    return next(module.parameters()).device

def lens_logprobs(h_last):
    """Logit lens: final norm + LM head on a (1, d) hidden state -> (vocab,) fp32 log-probs."""
    z = LM.model.norm(h_last.to(_dev(LM.model.norm)))
    return LM.lm_head(z.to(_dev(LM.lm_head))).float().log_softmax(-1)[0]

@torch.no_grad()
def rows_attention(layer, h, pos_ids, rows, img_s, img_n):
    """Recompute layer attention for a few query rows only (no full attention matrix).
    Returns per row: visual attention mass, and the size of the change in the attention update when the
    visual keys are removed, relative to the residual stream (rel_res) and to the attention output (rel_attn)."""
    dev = _dev(layer)
    attn = layer.self_attn
    h = h.to(dev)
    pos_ids = pos_ids.to(dev)
    rows_t = torch.as_tensor(rows, device=dev)
    x = layer.input_layernorm(h)                                            # (1, n, d)
    n, r = x.shape[1], len(rows)
    H, Hkv, hd = attn.num_heads, attn.num_key_value_heads, attn.head_dim
    q = attn.q_proj(x[:, rows_t]).view(1, r, H, hd).transpose(1, 2)        # (1, H, r, hd)
    k = attn.k_proj(x).view(1, n, Hkv, hd).transpose(1, 2)
    v = attn.v_proj(x).view(1, n, Hkv, hd).transpose(1, 2)
    cos, sin = attn.rotary_emb(v, pos_ids)                                  # (1, n, hd)
    q = q * cos[:, rows_t].unsqueeze(1) + rotate_half(q) * sin[:, rows_t].unsqueeze(1)
    k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    k, v = repeat_kv(k, attn.num_key_value_groups), repeat_kv(v, attn.num_key_value_groups)
    s = (q.float() @ k.float().transpose(2, 3)) / math.sqrt(hd)            # (1, H, r, n)
    key_pos = torch.arange(n, device=dev)
    s = s.masked_fill((key_pos[None, :] > rows_t[:, None])[None, None], float("-inf"))   # causal
    img = torch.zeros(n, dtype=torch.bool, device=dev)
    img[img_s:img_s + img_n] = True
    p = s.softmax(-1)
    p_txt = s.masked_fill(img[None, None, None, :], float("-inf")).softmax(-1)
    mass = p[..., img].sum(-1).mean(1)[0]                                   # (r,)
    vf = v.float()
    o_full = (p @ vf).transpose(1, 2).reshape(1, r, H * hd)
    o_diff = ((p - p_txt) @ vf).transpose(1, 2).reshape(1, r, H * hd)
    full = attn.o_proj(o_full.to(x.dtype)).float()[0]
    delta = attn.o_proj(o_diff.to(x.dtype)).float()[0]
    resid = h[0, rows_t].float()
    rel_res = delta.norm(dim=-1) / resid.norm(dim=-1).clamp_min(1e-6)
    rel_attn = delta.norm(dim=-1) / full.norm(dim=-1).clamp_min(1e-6)
    return mass, rel_res, rel_attn

def _run_layer(layer, h, mask, pos, cpos):
    return layer(h, attention_mask=mask, position_ids=pos, past_key_value=None,
                 output_attentions=False, use_cache=False, cache_position=cpos)[0]

@torch.no_grad()
def profile_forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                    inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None,
                    return_dict=None, cache_position=None, fastv_config=None):
    cfg = fastv_config
    img_s, img_n = cfg["image_token_start_index"], cfg["image_token_length"]
    h = inputs_embeds
    n, dev0, L = h.shape[1], h.device, len(self.layers)
    assert n > img_s + img_n, "profile_forward is prefill-only"
    pos = torch.arange(n, device=dev0)[None]
    cpos = torch.arange(n, device=dev0)
    mask = self._update_causal_mask(attention_mask, h, 0)
    text_idx = torch.cat((torch.arange(img_s, device=dev0), torch.arange(img_s + img_n, n, device=dev0)))
    pos_t, cpos_t = text_idx[None], torch.arange(text_idx.numel(), device=dev0)
    rows = list(range(img_s + img_n, n))                                     # question + "ASSISTANT:" tokens

    sig = defaultdict(list)
    exit_lp, lens_lp = [], []
    for E in range(L + 1):
        # Everything indexed by E is available before layer E runs.
        lens_lp.append(lens_logprobs(h[:, -1]))
        if E == L:
            break
        mass, rel_res, rel_attn = rows_attention(self.layers[E], h, pos, rows, img_s, img_n)
        sig["mass_last"].append(mass[-1].item());      sig["mass_text"].append(mass.mean().item())
        sig["cf_last_res"].append(rel_res[-1].item()); sig["cf_text_res"].append(rel_res.mean().item())
        sig["cf_last_attn"].append(rel_attn[-1].item()); sig["cf_text_attn"].append(rel_attn.mean().item())
        # Exit branch: drop all visual tokens now, run layers E..L-1 on text only.
        ht = h[:, text_idx.to(h.device)]
        mask_t = self._update_causal_mask(None, ht, 0)
        for j in range(E, L):
            ht = _run_layer(self.layers[j], ht, mask_t, pos_t, cpos_t)
        exit_lp.append(lens_logprobs(ht[:, -1]))
        # Continue the full-sequence pass.
        h = _run_layer(self.layers[E], h, mask, pos, cpos)

    base_lp = lens_lp[L]                                  # = the model's true final next-token distribution
    base_p = base_lp.exp()
    base_top1 = int(base_lp.argmax())
    exit_top1 = [int(lp.argmax()) for lp in exit_lp] + [base_top1]
    PROFILE.clear()
    PROFILE.update(
        n=n, base_top1=base_top1, base_tok=tokenizer.decode([base_top1]),
        preserved=[t == base_top1 for t in exit_top1],                       # length L+1 (E=0..L)
        exit_top1=exit_top1,
        exit_kl=[float((base_p * (base_lp - lp.to(base_lp.device))).sum()) for lp in exit_lp] + [0.0],
        lens_top1=[int(lp.argmax()) for lp in lens_lp],
        lens_p1=[float(lp.max().exp()) for lp in lens_lp],
        lens_p_base=[float(lp[base_top1].exp()) for lp in lens_lp],
        lens_kl_prev=[float("nan")] + [float((lens_lp[e].exp() * (lens_lp[e] - lens_lp[e - 1].to(lens_lp[e].device))).sum())
                                       for e in range(1, L + 1)],
        **{k: v for k, v in sig.items()},
    )
    hs = self.norm(h)
    if not return_dict:
        return (hs,)
    return BaseModelOutputWithPast(last_hidden_state=hs, past_key_values=None, hidden_states=None, attentions=None)

# ---- dispatcher (install once; safe to re-run) ----
_LMODEL = modeling_llama.LlamaModel
if not hasattr(_LMODEL, "_fastv_forward_original"):
    _LMODEL._fastv_forward_original = _LMODEL.fastv_forward
assert getattr(_LMODEL._fastv_forward_original, "__module__", "") == modeling_llama.__name__, \
    "fastv_forward already patched by another notebook in this kernel: restart and run from the top."
METHODS = {"profile": lambda *a, **k: profile_forward(*a, **k)}
def _fastv_dispatch(self, *args, fastv_config=None, **kwargs):
    fn = METHODS.get((fastv_config or {}).get("method"))
    if fn is not None:
        return fn(self, *args, fastv_config=fastv_config, **kwargs)
    return _LMODEL._fastv_forward_original(self, *args, fastv_config=fastv_config, **kwargs)
_LMODEL.fastv_forward = _fastv_dispatch

PROFILE_CFG = {"method": "profile", "image_token_start_index": IMAGE_START, "image_token_length": IMAGE_LEN}
def fastv_cfg(k, r):
    return {"method": "fastv", "use_fastv": True, "fastv_k": k, "fastv_r": r,
            "image_token_start_index": IMAGE_START, "image_token_length": IMAGE_LEN}

@torch.no_grad()
def profile(inputs):
    set_method(PROFILE_CFG)
    try:
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
    finally:
        set_method(None)
    return dict(PROFILE), logits
print("profiler installed")
''')

md(r'''
## 6. Data — same sampling as Phase 1, larger subsets
POPE: 300 questions per split (900 total; superset of the Phase 1 subset). GQA: 600 testdev-balanced questions
(superset of the Phase 1 300).
''')
code(r'''
%%bash
mkdir -p /kaggle/working/data && cd /kaggle/working/data
[ -d POPE ] || git clone --depth 1 https://github.com/AoiDragon/POPE.git
''')
code(r'''
N_PER_SPLIT, N_GQA = 300, 600
GQA_SUFFIX = "\nAnswer the question using a single word or phrase."

# ---- POPE ----
pope_dir = DATA / "POPE/output/coco"
coco_dir = DATA / "coco_val2014_subset"; coco_dir.mkdir(exist_ok=True)
random.seed(0)
pope_items = []
for split in ["coco_pope_random.json", "coco_pope_popular.json", "coco_pope_adversarial.json"]:
    lines = [json.loads(l) for l in open(pope_dir / split) if l.strip()]
    images = sorted(set(l["image"] for l in lines))
    random.shuffle(images)
    keep = set(images[: max(1, N_PER_SPLIT // 6)])
    for l in [l for l in lines if l["image"] in keep][:N_PER_SPLIT]:
        pope_items.append({"id": f"{split.split('_')[2].split('.')[0]}:{l['question_id']}",
                           "image_path": str(coco_dir / l["image"]), "question": l["text"], "label": l["label"]})
def fetch(fname):
    dest = coco_dir / fname
    if not dest.exists():
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2014/{fname}", dest)
with ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(fetch, sorted({Path(it["image_path"]).name for it in pope_items})))
print("POPE:", len(pope_items), "questions")

# ---- GQA ----
GQA_Q_FILE = next(Path("/kaggle/input").rglob("testdev_balanced_questions.json"))
gqa_q = json.load(open(GQA_Q_FILE))
def find_image_dir(sample_image_id, root="/kaggle/input"):
    for dirpath, dirnames, filenames in os.walk(root):
        if f"{sample_image_id}.jpg" in filenames:
            return Path(dirpath)
        if len(filenames) > 1000:
            dirnames[:] = []
    raise FileNotFoundError("GQA images not found under /kaggle/input")
qids = list(gqa_q.keys())
random.Random(0).shuffle(qids)
GQA_IMG_DIR = find_image_dir(gqa_q[qids[0]]["imageId"])
gqa_items = [{"id": q, "image_path": str(GQA_IMG_DIR / f"{gqa_q[q]['imageId']}.jpg"),
              "question": gqa_q[q]["question"], "answer": gqa_q[q]["answer"],
              "structural": gqa_q[q].get("types", {}).get("structural")} for q in qids[:N_GQA]]
missing = [it for it in gqa_items if not Path(it["image_path"]).exists()]
assert not missing, f"{len(missing)} GQA images missing"
print("GQA:", len(gqa_items), "questions from", GQA_IMG_DIR)

# bench -> (items, prompt suffix, answer kind)
BENCH = {"pope": (pope_items, "", "yesno"), "gqa": (gqa_items, GQA_SUFFIX, "open")}
''')

md(r'''
## 6b. TextVQA (500) and MME (≤50 per category) from the HF hub
TextVQA needs reading text in the image, so its visual dependency should run deep. MME covers 14 task types.
Images are saved to disk so the profiler loop stays unchanged. Skip this cell if the download is too slow; the
rest of the notebook works with whatever benchmarks are present in `BENCH`.
''')
code(r'''
from datasets import load_dataset

N_TEXTVQA, N_MME_PER_CAT = 500, 50
SINGLE_WORD = "\nAnswer the question using a single word or phrase."

def save_img(img, path):
    if not path.exists():
        img.convert("RGB").save(path, format="JPEG", quality=95)

# ---- TextVQA (validation split; VQA-style 10 answers per question) ----
tv_dir = DATA / "textvqa"; tv_dir.mkdir(exist_ok=True)
tv = load_dataset("lmms-lab/textvqa", split="validation").shuffle(seed=0).select(range(N_TEXTVQA))
textvqa_items = []
for ex in tv:
    path = tv_dir / f"{ex['question_id']}.jpg"
    save_img(ex["image"], path)
    textvqa_items.append({"id": f"tv:{ex['question_id']}", "image_path": str(path),
                          "question": ex["question"], "answer": list(ex["answers"])})
print("TextVQA:", len(textvqa_items))

# ---- MME (stratified by category) ----
mme_dir = DATA / "mme"; mme_dir.mkdir(exist_ok=True)
try:
    mme = load_dataset("lmms-lab/MME", split="test")
except Exception as e:
    print("lmms-lab/MME failed:", e); mme = load_dataset("lmms-lab-encoder/MME", split="test")
by_cat = defaultdict(list)
for i, ex in enumerate(mme):
    by_cat[ex["category"]].append(i)
rng = random.Random(0)
mme_items = []
for cat, idxs in sorted(by_cat.items()):
    for i in sorted(rng.sample(idxs, min(N_MME_PER_CAT, len(idxs)))):
        ex = mme[i]
        path = mme_dir / f"{cat}_{ex['question_id'].replace('/', '_')}.jpg"
        save_img(ex["image"], path)
        mme_items.append({"id": f"mme:{cat}:{ex['question_id']}", "image_path": str(path), "category": cat,
                          "question": ex["question"], "label": ex["answer"].strip().lower()})
print("MME:", len(mme_items), "over", len(by_cat), "categories")

BENCH["textvqa"] = (textvqa_items, SINGLE_WORD, "open")   # MME questions already end with "Please answer yes or no."
BENCH["mme"] = (mme_items, "", "yesno")
''')

md(r'''
## 6c. Restore already-finished profiles (optional)
If you attached `phaseA_results.zip`'s contents as a Kaggle dataset, this copies `profile_*.jsonl` / `verify_*.jsonl`
into `/kaggle/working/phaseA` so POPE and GQA are not re-profiled.
''')
code(r'''
import shutil
restored = 0
for p in Path("/kaggle/input").rglob("profile_*.jsonl"):
    dest = OUT / p.name
    if not dest.exists():
        shutil.copy(p, dest); restored += 1
for p in Path("/kaggle/input").rglob("verify_*.jsonl"):
    dest = OUT / p.name
    if not dest.exists():
        shutil.copy(p, dest); restored += 1
print("restored files:", restored, "| present:", sorted(x.name for x in OUT.glob("*.jsonl")))
''')

md(r'''
## 7a. Smoke test — must pass before the long run
Checks: (1) the profiler's final logits match the stock forward; (2) the exit branch at E=0 equals a text-only run;
(3) the signals look sane. Expect max |logit diff| ≲ 0.05 in fp16.
''')
code(r'''
it = pope_items[0]
inputs = build_inputs(Image.open(it["image_path"]).convert("RGB"), it["question"])
with torch.no_grad():
    stock = model(**inputs, use_cache=False).logits[0, -1].float()
t0 = time.time()
prof, logits = profile(inputs)
print(f"profile time: {time.time() - t0:.1f}s | seq len {prof['n']}")
print("max |logit diff| profiler vs stock:", (logits - stock).abs().max().item())
print("baseline first token:", repr(prof["base_tok"]), "| question:", it["question"], "| label:", it["label"])
print("preserved[E] for E=0..32:", "".join("1" if x else "0" for x in prof["preserved"]))
print("\n E   mass_last  cf_last_res  cf_text_res  lens_p1  lens_p_base")
for E in range(0, 32, 3):
    print(f"{E:2d}   {prof['mass_last'][E]:.3f}      {prof['cf_last_res'][E]:.4f}       {prof['cf_text_res'][E]:.4f}"
          f"     {prof['lens_p1'][E]:.3f}    {prof['lens_p_base'][E]:.3f}")
assert prof["preserved"][32], "E=32 must equal the baseline"
assert (logits - stock).abs().max().item() < 0.5, "profiler disagrees with the stock forward -- stop and report"
''')

md(r'''
## 7b. Profiling run (resumable; ~2 s/example → ~50 min for 1500 examples)
Writes one JSON line per example. Re-running skips finished examples. Zips results at the end.
''')
code(r'''
def profile_bench(bench, max_items=None):
    items, suffix, _kind = BENCH[bench]
    items = items[:max_items] if max_items else items
    path = OUT / f"profile_{bench}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    todo = [it for it in items if it["id"] not in done]
    print(f"[{bench}] {len(done)} done, {len(todo)} to run")
    t0 = time.time()
    with open(path, "a") as f:
        for i, it in enumerate(todo):
            inputs = build_inputs(Image.open(it["image_path"]).convert("RGB"), it["question"], suffix)
            prof, _ = profile(inputs)
            rec = {"id": it["id"], "bench": bench, "label": it.get("label"), "answer": it.get("answer"),
                   "structural": it.get("structural"), "category": it.get("category"), **prof}
            rec = {k: ([round(x, 6) if isinstance(x, float) else x for x in v] if isinstance(v, list) else v)
                   for k, v in rec.items()}
            f.write(json.dumps(rec) + "\n"); f.flush()
            if (i + 1) % 50 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(todo)}  ({el/(i+1):.2f}s/ex, ~{el/(i+1)*(len(todo)-i-1)/60:.0f} min left)")
            if (i + 1) % 200 == 0:
                free_mem()
    free_mem()

for bench in BENCH:
    profile_bench(bench)
os.system(f"cd {WORK} && zip -rq phaseA_results.zip phaseA")
print("zipped ->", WORK / "phaseA_results.zip")
''')

md(r'''
## 7c. Verification — is "same first token" a faithful proxy for "same answer"?
For a random subset, generate the full answer with all visual tokens dropped at each example's earliest safe exit
layer (FastV with R=1.0 drops all 576 image tokens after K=E layers) and compare it with the full model's answer.
Expect ≥95% agreement; if GQA is much lower, the analysis must use full-answer agreement instead.
''')
code(r'''
N_VERIFY = 100
def verify_bench(bench):
    items, suffix, kind = BENCH[bench]
    by_id = {it["id"]: it for it in items}
    profs = [json.loads(l) for l in open(OUT / f"profile_{bench}.jsonl")]
    rng = random.Random(0)
    sample = rng.sample(profs, min(N_VERIFY, len(profs)))
    path = OUT / f"verify_{bench}.jsonl"
    done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
    mnt = 3 if kind == "yesno" else 10
    with open(path, "a") as f:
        for p in sample:
            if p["id"] in done:
                continue
            E = next(e for e, ok in enumerate(p["preserved"]) if ok)          # earliest safe exit layer
            if E == 0 or E == 32:
                continue                                                     # FastV needs 1 <= K <= 31
            it = by_id[p["id"]]
            inputs = build_inputs(Image.open(it["image_path"]).convert("RGB"), it["question"], suffix)
            set_method(None); full = generate(inputs, mnt)
            set_method(fastv_cfg(E, 1.0)); ex = generate(inputs, mnt); set_method(None)
            same = (parse_yes_no(full) == parse_yes_no(ex)) if kind == "yesno" else (norm_answer(full) == norm_answer(ex))
            f.write(json.dumps({"id": p["id"], "E": E, "full": full, "exit": ex, "same": same}) + "\n"); f.flush()
    recs = [json.loads(l) for l in open(path)]
    print(f"[{bench}] full-answer agreement at the first-token oracle exit: "
          f"{np.mean([r['same'] for r in recs])*100:.1f}% over {len(recs)} examples")
    free_mem()

for bench in BENCH:
    verify_bench(bench)
os.system(f"cd {WORK} && zip -rq phaseA_results.zip phaseA")
''')

md(r'''
## 8. Analysis — the go/no-go test (CPU only; can run in a fresh session from the JSONL files)
Set `PHASEA_DIR` to wherever `profile_*.jsonl` live (e.g. an attached dataset).
''')
code(r'''
import json, math
from collections import Counter
from pathlib import Path
import numpy as np

PHASEA_DIR = Path("/kaggle/working/phaseA")
N_LAYERS, D_MODEL, D_FFN, IMAGE_LEN = 32, 4096, 11008, 576

def layer_flops(n):
    return 4 * n * D_MODEL**2 + 2 * n * n * D_MODEL + 2 * n * D_MODEL * D_FFN

def flops_table(ns):
    """(N, 33) relative prefill FLOPs for exit at E = 0..32."""
    ns = np.asarray(ns, dtype=float)
    E = np.arange(N_LAYERS + 1)[None, :]
    full, txt = layer_flops(ns)[:, None], layer_flops(ns - IMAGE_LEN)[:, None]
    return (E * full + (N_LAYERS - E) * txt) / (N_LAYERS * full)

SIGNALS = {   # name -> (array key, direction): exit when value <= tau ("low") or >= tau ("high")
    "cf_last_res": ("cf_last_res", "low"), "cf_text_res": ("cf_text_res", "low"),
    "cf_last_attn": ("cf_last_attn", "low"), "cf_text_attn": ("cf_text_attn", "low"),
    "mass_last": ("mass_last", "low"), "mass_text": ("mass_text", "low"),
    "lens_kl_prev": ("lens_kl_prev", "low"), "lens_p1": ("lens_p1", "high"),
}

def load(bench):
    recs = [json.loads(l) for l in open(PHASEA_DIR / f"profile_{bench}.jsonl")]
    D = {"ids": [r["id"] for r in recs], "recs": recs,
         "pres": np.array([r["preserved"] for r in recs], dtype=bool),            # (N, 33)
         "flops": flops_table([r["n"] for r in recs])}
    for name, (key, _) in SIGNALS.items():
        D[name] = np.array([r[key][:N_LAYERS] for r in recs], dtype=float)       # (N, 32): E = 0..31
    return D

DATASETS = {p.stem.split("_", 1)[1]: load(p.stem.split("_", 1)[1]) for p in sorted(PHASEA_DIR.glob("profile_*.jsonl"))}
for b, D in DATASETS.items():
    print(f"{b}: {len(D['ids'])} profiled examples")
''')
code(r'''
# 8.1 Oracle exit layers and headroom vs a fixed exit layer
def earliest_safe(pres):
    return pres.argmax(1)                                   # first True (E=32 is always True)

def earliest_stable(pres):
    tail_ok = np.flip(np.cumprod(np.flip(pres, 1), 1), 1).astype(bool)   # all preserved from E onwards
    return tail_ok.argmax(1)

for b, D in DATASETS.items():
    pres, fl = D["pres"], D["flops"]
    N = len(pres)
    es, est = earliest_safe(pres), earliest_stable(pres)
    rows = np.arange(N)
    print(f"\n==== {b.upper()} (n={N}) ====")
    if b == "pope":
        acc = np.mean([("yes" if r["base_tok"].strip().lower().startswith("yes") else "no") == r["label"] for r in D["recs"]])
        print(f"baseline POPE accuracy from first token: {acc*100:.1f}%")
    print(f"answer unchanged with NO image at all (E=0): {pres[:, 0].mean()*100:.1f}%")
    print(f"oracle earliest-safe E: median {np.median(es):.0f}, mean {es.mean():.1f}  -> FLOPs {fl[rows, es].mean()*100:.1f}%")
    print(f"oracle earliest-stable E: median {np.median(est):.0f}, mean {est.mean():.1f} -> FLOPs {fl[rows, est].mean()*100:.1f}%")
    print("stable-E histogram:", dict(sorted(Counter(est.tolist()).items())))
    print("fixed exit layer E (VTW-style): fidelity / FLOPs")
    print("  " + "  ".join(f"E{e}:{pres[:, e].mean()*100:.1f}/{fl[:, e].mean()*100:.0f}" for e in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28)))
    for T in (0.99, 0.98, 0.95):
        ok = [e for e in range(N_LAYERS + 1) if pres[:, e].mean() >= T]
        print(f"  smallest fixed E with fidelity >= {T:.0%}: E={ok[0]} @ {fl[:, ok[0]].mean()*100:.1f}% FLOPs")
''')
code(r'''
# 8.2 Per-layer AUC: does the signal at layer E predict whether exiting at E is safe?
def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if pos.size < 10 or neg.size < 10:
        return float("nan")
    allv = np.concatenate([pos, neg])   # tie-aware average ranks (Mann-Whitney)
    order = allv.argsort(); sv = allv[order]; r = np.empty_like(allv)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))

for b, D in DATASETS.items():
    print(f"\n==== {b.upper()}: AUC of 'safe to exit at E' (0.5 = chance; layers with >=10 of each class) ====")
    print(f"{'signal':<14}" + "".join(f"{'E'+str(e):>7}" for e in range(2, 30, 3)))
    for name, (_, direction) in SIGNALS.items():
        S = D[name]; row = []
        for e in range(2, 30, 3):
            x = -S[:, e] if direction == "low" else S[:, e]            # higher x = predicts "safe"
            safe = D["pres"][:, e]
            row.append(auc(x[safe], x[~safe]))
        print(f"{name:<14}" + "".join(f"{a:>7.3f}" for a in row))
''')
code(r'''
# 8.3 Policies + cross-validated go/no-go against the best fixed exit layer
def policy_exit(S, direction, tau, m, e_min):
    """First E >= e_min where the signal has satisfied the condition for m consecutive layers; else 32 (no exit)."""
    cond = (S <= tau) if direction == "low" else (S >= tau)
    cond = np.nan_to_num(cond, nan=False).astype(int)
    run = np.zeros_like(cond)
    for e in range(cond.shape[1]):
        run[:, e] = (run[:, e - 1] + 1) * cond[:, e] if e > 0 else cond[:, e]
    ok = run >= m
    ok[:, :e_min] = False
    ok = np.concatenate([ok, np.ones((ok.shape[0], 1), bool)], 1)   # E=32 fallback
    return ok.argmax(1)

def evaluate(E, pres, fl):
    r = np.arange(len(E))
    return pres[r, E].mean(), fl[r, E].mean()

def configs(S, direction):
    finite = S[np.isfinite(S)]
    taus = np.unique(np.quantile(finite, np.linspace(0.02, 0.98, 25)))
    return [(t, m, e) for t in taus for m in (1, 2, 3) for e in (1, 2, 4, 8)]

TARGETS = (0.99, 0.98, 0.95)
N_SPLITS = 20
summary = {}
for b, D in DATASETS.items():
    pres, fl, N = D["pres"], D["flops"], len(D["pres"])
    rng = np.random.default_rng(0)
    splits = [rng.permutation(N) for _ in range(N_SPLITS)]
    print(f"\n==== {b.upper()}: held-out (2-fold, {N_SPLITS} random splits) fidelity / FLOPs at matched target ====")
    print(f"{'method':<16}" + "".join(f"{'T='+str(int(T*100))+'%':>22}" for T in TARGETS))
    # cache all policy exits once
    cache = {name: {c: policy_exit(D[name], d, *c) for c in configs(D[name], d)} for name, (_, d) in SIGNALS.items()}
    results = {}
    for method in ["fixed_E"] + list(SIGNALS):
        cells = []
        for T in TARGETS:
            fids, fls = [], []
            for perm in splits:
                A, B = perm[: N // 2], perm[N // 2:]
                for tr, te in ((A, B), (B, A)):
                    if method == "fixed_E":
                        e = next(e for e in range(N_LAYERS + 1) if pres[tr, e].mean() >= T)
                        Ete = np.full(len(te), e)
                    else:
                        best = None
                        for c, Eall in cache[method].items():
                            f_tr, c_tr = evaluate(Eall[tr], pres[tr], fl[tr])
                            if f_tr >= T and (best is None or c_tr < best[0]):
                                best = (c_tr, c)
                        Ete = cache[method][best[1]][te] if best else np.full(len(te), N_LAYERS)
                    f_te, c_te = evaluate(Ete, pres[te], fl[te])
                    fids.append(f_te); fls.append(c_te)
            results[(method, T)] = (np.mean(fids), np.mean(fls), np.array(fls))
            cells.append(f"{np.mean(fids)*100:6.1f}% / {np.mean(fls)*100:5.1f}%")
        print(f"{method:<16}" + "".join(f"{c:>22}" for c in cells))
    summary[b] = results

    # shuffled control for the best signal at T=98%: same exit-layer distribution, random assignment
    best_sig = min(SIGNALS, key=lambda s: results[(s, 0.98)][1] if results[(s, 0.98)][0] >= 0.975 else 9)
    c_best = None
    for c, Eall in cache[best_sig].items():
        f, cst = evaluate(Eall, pres, fl)
        if f >= 0.98 and (c_best is None or cst < c_best[0]):
            c_best = (cst, c)
    if c_best:
        Eall = cache[best_sig][c_best[1]]
        f_sig, c_sig = evaluate(Eall, pres, fl)
        f_shuf = np.mean([evaluate(np.random.default_rng(s).permutation(Eall), pres, fl)[0] for s in range(50)])
        tau, m, e_min = c_best[1]
        print(f"\nbest signal at T=98%: {best_sig} (tau={float(tau):.4g}, m={m}, E_min={e_min}) -> in-sample fidelity {f_sig*100:.1f}% @ {c_sig*100:.1f}% FLOPs;"
              f" shuffled exits (same distribution): fidelity {f_shuf*100:.1f}%")
''')
code(r'''
# 8.5 One global threshold for ALL benchmarks (the realistic deployment setting) vs one fixed exit layer for all.
# Thresholds are chosen on half of the pooled data and evaluated on the other half, per benchmark (2-fold x 10).
names = list(DATASETS)
pres_all = np.concatenate([DATASETS[b]["pres"] for b in names])
fl_all = np.concatenate([DATASETS[b]["flops"] for b in names])
bench_of = np.concatenate([[b] * len(DATASETS[b]["pres"]) for b in names])
sig_all = {s: np.concatenate([DATASETS[b][s] for b in names]) for s in SIGNALS}
T_POOL = 0.98
rng = np.random.default_rng(0)
print(f"==== POOLED calibration at T={T_POOL:.0%}: held-out fidelity / FLOPs per benchmark ====")
print(f"{'method':<14}" + "".join(f"{b:>18}" for b in names))
pooled = {}
for method in ["fixed_E"] + list(SIGNALS):
    per_b = {b: [] for b in names}
    if method != "fixed_E":
        d = SIGNALS[method][1]
        cands = {c: policy_exit(sig_all[method], d, *c) for c in configs(sig_all[method], d)}
    for _ in range(10):
        perm = rng.permutation(len(pres_all)); A, B = perm[: len(perm) // 2], perm[len(perm) // 2:]
        for tr, te in ((A, B), (B, A)):
            if method == "fixed_E":
                e = next(e for e in range(N_LAYERS + 1) if pres_all[tr, e].mean() >= T_POOL)
                Ete = np.full(len(te), e)
            else:
                best = None
                for c, Eall in cands.items():
                    f, cst = evaluate(Eall[tr], pres_all[tr], fl_all[tr])
                    if f >= T_POOL and (best is None or cst < best[0]):
                        best = (cst, c)
                Ete = cands[best[1]][te] if best else np.full(len(te), N_LAYERS)
            r = np.arange(len(te))
            for b in names:
                m = bench_of[te] == b
                if m.any():
                    per_b[b].append((pres_all[te][r, Ete][m].mean(), fl_all[te][r, Ete][m].mean()))
    pooled[method] = {b: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for b, v in per_b.items()}
    print(f"{method:<14}" + "".join(f"{pooled[method][b][0]*100:6.1f}%/{pooled[method][b][1]*100:5.1f}%   " for b in names))

print("\n==== Cross-benchmark transfer at T=98%: calibrate on one benchmark, apply to the others ====")
for src in names:
    D = DATASETS[src]
    e_fix = next(e for e in range(N_LAYERS + 1) if D["pres"][:, e].mean() >= 0.98)
    for method in ("fixed_E", "lens_kl_prev", "cf_last_res"):
        if method != "fixed_E":
            d = SIGNALS[method][1]; best = None
            for c in configs(sig_all[method], d):
                f, cst = evaluate(policy_exit(D[method], d, *c), D["pres"], D["flops"])
                if f >= 0.98 and (best is None or cst < best[0]):
                    best = (cst, c)
        row = []
        for dst in names:
            Dd = DATASETS[dst]
            E = np.full(len(Dd["pres"]), e_fix) if method == "fixed_E" else (
                policy_exit(Dd[method], SIGNALS[method][1], *best[1]) if best else np.full(len(Dd["pres"]), N_LAYERS))
            f, c = evaluate(E, Dd["pres"], Dd["flops"])
            row.append(f"{f*100:5.1f}%/{c*100:5.1f}%")
        print(f"  from {src:<8} {method:<13}" + "".join(f"{x:>16}" for x in row))
''')

code(r'''
# 8.4 Verdict
print("GO / NO-GO (held-out, vs best fixed exit layer; FLOPs saving in percentage points, fidelity within 0.5 pt of target)")
for b, R in summary.items():
    print(f"\n{b.upper()}")
    for T in TARGETS:
        f_fix, c_fix, _ = R[("fixed_E", T)]
        best = None
        for s in SIGNALS:
            f, c, arr = R[(s, T)]
            if f >= T - 0.005 and (best is None or c < best[1]):
                best = (s, c, f)
        if best:
            print(f"  T={T:.0%}: fixed E -> {c_fix*100:.1f}% FLOPs (fid {f_fix*100:.1f}%) | best signal {best[0]} -> "
                  f"{best[1]*100:.1f}% FLOPs (fid {best[2]*100:.1f}%) | saving {(c_fix-best[1])*100:+.1f} pt")
        else:
            print(f"  T={T:.0%}: no signal meets the target on held-out data")
print("\nPASS if a signal saves >= ~5 pt FLOPs at T=98% on BOTH benchmarks and beats its shuffled control.")
''')

nb = {"cells": [({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)} if t == "markdown" else
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open("../../notebooks/phaseA-exit-profiling-v2.ipynb", "w"), indent=1)
print(len(cells), "cells")
