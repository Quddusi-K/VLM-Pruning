import json

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# Phase B — kernel-compatible inference engine with per-image visual-token exit

A prefill + greedy-decode loop over LLaVA-1.5-7B's decoder layers that:
- runs on **SDPA attention with a KV cache** (never materializes an attention matrix);
- can drop **all** visual tokens at a layer chosen **per image** by a cheap signal (logit-lens stability or a 1-row
  counterfactual probe), or at a fixed layer (VTW-style), or not at all (baseline);
- optionally keeps only a **spatially uniform** subset of visual tokens from an early layer on (Phase C);
- records exit layer, exact per-layer sequence lengths (→ FLOPs), prefill / decode / end-to-end latency.

**Transformers version.** Two supported setups, pick one with `USE_FASTV_FORK` in §2:
- `False` (default): fresh Kaggle notebook, stock `transformers==4.49.0`. Processor expands `<image>` into 576 tokens itself.
- `True`: the same FastV fork (4.39) you used for Phase A. Useful if 4.49 fails to install, and it lets you time FastV (eager) on
  the same model later.

The engine is written to work with both layouts (it checks the module structure at runtime).

**Run order:** §1–§4, then **§5 unit tests must pass** (engine == stock model), then §6 (threshold calibration from
Phase A), §7 latency, §8 evaluation. If §5 fails, stop and send the traceback + printed versions.
''')

md("## 1. Environment")
code(r'''
import subprocess, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
''')

md("## 2. Install (pick ONE path, then restart the kernel if `transformers` was already imported)")
code(r'''
USE_FASTV_FORK = False
TRANSFORMERS_PIN = "4.49.0"

import subprocess, sys
if USE_FASTV_FORK:
    subprocess.run("""
    cd /kaggle/working && ([ -d FastV ] || git clone --depth 1 https://github.com/pkunlp-icler/FastV.git)
    cd FastV/src/FastV/llava-hf/transformers && pip install -e . --no-deps -q
    pip install -q "tokenizers<0.19,>=0.14" pillow accelerate sentencepiece datasets
    """, shell=True, check=True)
else:
    subprocess.run(f"pip install -q transformers=={TRANSFORMERS_PIN} accelerate sentencepiece pillow datasets", shell=True, check=True)
print(subprocess.run([sys.executable, "-c", "import transformers; print('transformers', transformers.__version__)"],
                     capture_output=True, text=True).stdout)
''')

md("## 3. Imports, constants, model (fp16, SDPA, split across GPUs)")
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
from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor, LlavaProcessor, LlavaForConditionalGeneration
from transformers.cache_utils import DynamicCache
import transformers.models.llama.modeling_llama as modeling_llama
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv

print("transformers", transformers.__version__)
USE_FASTV_FORK = "fastv_forward" in dir(modeling_llama.LlamaModel)

WORK = Path("/kaggle/working"); DATA, OUT = WORK / "data", WORK / "phaseB"
for p in (DATA, OUT):
    p.mkdir(parents=True, exist_ok=True)
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
IMAGE_LEN, GRID = 576, 24
N_LAYERS, D_MODEL, D_FFN = 32, 4096, 11008

if USE_FASTV_FORK:   # 4.39 cannot parse the updated processor config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    processor = LlavaProcessor(image_processor=CLIPImageProcessor.from_pretrained(MODEL_ID), tokenizer=tokenizer)
else:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    tokenizer = processor.tokenizer

if "model" not in globals():
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
    model.eval()
if USE_FASTV_FORK:
    model.config.fastv_config = None

# ---- layout-robust accessors ----
_lm = model.language_model
if hasattr(_lm, "model") and hasattr(_lm, "lm_head"):      # LlamaForCausalLM (transformers <= 4.51)
    CORE, LM_HEAD = _lm.model, _lm.lm_head
else:                                                      # LlamaModel (>= 4.52); lm_head on the top-level model
    CORE, LM_HEAD = _lm, model.lm_head
LAYERS, NORM, EMBED = CORE.layers, CORE.norm, CORE.embed_tokens
ROTARY = getattr(CORE, "rotary_emb", None)                 # model-level rotary (>= 4.45); None on the 4.39 fork
IMG_TOK = getattr(model.config, "image_token_index", None) or model.config.image_token_id
EOS = tokenizer.eos_token_id
attn_cls = type(LAYERS[0].self_attn).__name__
attn_impl = getattr(CORE.config, "_attn_implementation", "?")
print(f"layers={len(LAYERS)} | attention class={attn_cls} | impl={attn_impl} | model rotary={ROTARY is not None}")
assert "sdpa" in attn_impl.lower() or "Sdpa" in attn_cls, "SDPA attention not active -- do not continue"
print("Devices:", sorted({str(p.device) for p in model.parameters()}))
''')

md(r'''
## 4. The engine
Merged embeddings (text + projected image features) are captured from the model's own forward with a pre-hook on the
language model, so the image-merging code path is the library's, whatever the version. The layer loop is ours.
''')
code(r'''
def _dev(m):
    return next(m.parameters()).device

# ---- capture merged inputs_embeds from the library's own LLaVA forward ----
class _Captured(Exception):
    pass
_CAP = {}
def _capture_hook(module, args, kwargs):
    emb = kwargs.get("inputs_embeds")
    _CAP["emb"] = emb if emb is not None else (args[0] if args else None)
    raise _Captured()

@torch.no_grad()
def merged_embeddings(inputs):
    h = model.language_model.register_forward_pre_hook(_capture_hook, with_kwargs=True)
    try:
        model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"),
              pixel_values=inputs["pixel_values"], use_cache=False)
    except _Captured:
        pass
    finally:
        h.remove()
    emb = _CAP.pop("emb")
    assert emb is not None and emb.dim() == 3 and emb.shape[0] == 1
    return emb

def visual_span(input_ids, n_embed):
    """(start, length) of the image features inside the merged sequence."""
    pos = (input_ids[0] == IMG_TOK).nonzero().flatten()
    assert pos.numel() >= 1, "no image token in input_ids"
    if pos.numel() >= IMAGE_LEN:                   # processor already expanded <image> into 576 tokens
        assert int(pos[-1] - pos[0]) == pos.numel() - 1, "image tokens are not contiguous"
        return int(pos[0]), int(pos.numel())
    assert n_embed == input_ids.shape[1] - 1 + IMAGE_LEN, (n_embed, input_ids.shape)
    return int(pos[0]), IMAGE_LEN

def build_inputs(image, question, suffix=""):
    prompt = f"USER: <image>\n{question}{suffix}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    return inputs

def uniform_visual_keep(ratio, grid=GRID):
    """Indices (0..575) of a spatially uniform subset of the 24x24 patch grid."""
    idx = torch.arange(grid * grid); r, c = idx // grid, idx % grid
    if ratio >= 1:
        return idx
    if abs(ratio - 0.5) < 1e-6:   m = (r + c) % 2 == 0
    elif abs(ratio - 0.25) < 1e-6: m = (r % 2 == 0) & (c % 2 == 0)
    elif abs(ratio - 0.75) < 1e-6: m = ~((r % 2 == 1) & (c % 2 == 1))
    else:
        k = round(grid * grid * ratio); m = torch.zeros(grid * grid, dtype=torch.bool)
        m[torch.linspace(0, grid * grid - 1, k).round().long()] = True
    return idx[m]

def lens_logprobs(h_last):
    """Logit lens: final norm + LM head on (1, d) -> (vocab,) fp32 log-probs."""
    z = NORM(h_last.to(_dev(NORM)))
    return LM_HEAD(z.to(_dev(LM_HEAD))).float().log_softmax(-1)[0]

def rope_for(layer, h, pos_ids):
    """(cos, sin) on the layer's device, or None when the attention module computes RoPE itself (4.39 fork)."""
    if ROTARY is None:
        return None
    dev = _dev(layer)
    cos, sin = ROTARY(h.to(dev), pos_ids.to(dev))
    return cos, sin

def run_layer(layer, h, pos_ids, cache, cache_pos):
    dev = _dev(layer)
    kw = dict(attention_mask=None, position_ids=pos_ids.to(dev), past_key_value=cache,
              output_attentions=False, use_cache=True, cache_position=cache_pos.to(dev))
    pe = rope_for(layer, h, pos_ids)
    if pe is not None:
        kw["position_embeddings"] = pe
    out = layer(h.to(dev), **kw)
    return out[0] if isinstance(out, (tuple, list)) else out

@torch.no_grad()
def cf_last_res(layer, h, pos_ids, vis_mask):
    """Counterfactual probe for the LAST token only: relative change of the attention update when the visual keys are
    removed. Recomputes one query row by hand; cost ~ one row of attention."""
    attn, dev = layer.self_attn, _dev(layer)
    x = layer.input_layernorm(h.to(dev)); n = x.shape[1]
    H = getattr(attn, "num_heads", None) or attn.config.num_attention_heads
    Hkv = getattr(attn, "num_key_value_heads", None) or attn.config.num_key_value_heads
    hd = attn.head_dim
    q = attn.q_proj(x[:, -1:]).view(1, 1, H, hd).transpose(1, 2)
    k = attn.k_proj(x).view(1, n, Hkv, hd).transpose(1, 2)
    v = attn.v_proj(x).view(1, n, Hkv, hd).transpose(1, 2)
    pe = rope_for(layer, h, pos_ids)
    cos, sin = pe if pe is not None else attn.rotary_emb(v, pos_ids.to(dev))
    cos, sin = cos.to(dev), sin.to(dev)
    q = q * cos[:, -1:].unsqueeze(1) + rotate_half(q) * sin[:, -1:].unsqueeze(1)
    k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    k, v = repeat_kv(k, H // Hkv), repeat_kv(v, H // Hkv)
    s = (q.float() @ k.float().transpose(2, 3)) / math.sqrt(hd)           # (1, H, 1, n); last row sees everything
    p = s.softmax(-1)
    p_txt = s.masked_fill(vis_mask.to(dev)[None, None, None, :], float("-inf")).softmax(-1)
    o = (((p - p_txt) @ v.float()).transpose(1, 2).reshape(1, 1, H * hd)).to(x.dtype)
    delta = attn.o_proj(o).float()[0, 0]
    return (delta.norm() / h[0, -1].to(dev).float().norm().clamp_min(1e-6)).item()


class ExitConfig:
    """exit: 'none' | 'fixed' | 'lens' | 'cf'.  lens/cf: exit at the first layer E >= e_min where the signal has met
    the threshold for m consecutive layers (lens: KL(lens_E || lens_{E-1}) <= tau; cf: cf_last_res <= tau).
    uniform_ratio < 1 keeps a uniform subset of visual tokens from layer uniform_layer onward (Phase C)."""
    def __init__(self, exit="none", exit_layer=None, tau=None, m=1, e_min=1, uniform_ratio=1.0, uniform_layer=2, name=None):
        self.exit, self.exit_layer, self.tau, self.m, self.e_min = exit, exit_layer, tau, m, e_min
        self.uniform_ratio, self.uniform_layer = uniform_ratio, uniform_layer
        self.name = name or self.auto_name()
    def auto_name(self):
        if self.exit == "none":
            s = "baseline"
        elif self.exit == "fixed":
            s = f"fixed_E{self.exit_layer}"
        else:
            s = f"{self.exit}_t{self.tau:.3g}_m{self.m}_e{self.e_min}"
        if self.uniform_ratio < 1:
            s += f"_uni{self.uniform_ratio}@L{self.uniform_layer}"
        return s


def _sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)

@torch.no_grad()
def engine_generate(inputs, cfg, max_new_tokens, stop_at_eos=True):
    """Prefill with optional uniform subsampling + per-image visual exit, then greedy decode with a KV cache.
    Returns dict(text, tokens, exit_layer, seq_lens, rel_flops, t_prefill, t_decode, t_total)."""
    _sync(); t0 = time.perf_counter()
    emb = merged_embeddings(inputs)
    n0 = emb.shape[1]
    img_s, img_n = visual_span(inputs["input_ids"], n0)
    dev0 = emb.device
    cur_pos = torch.arange(n0, device=dev0)                       # original position of every current token
    vis = torch.zeros(n0, dtype=torch.bool, device=dev0); vis[img_s:img_s + img_n] = True
    cache = DynamicCache()
    h = emb
    exit_layer, uniformed, prev_lp, run = None, False, None, 0
    seq_lens = []
    for E, layer in enumerate(LAYERS):
        # ---- Phase C: uniform subsampling of visual tokens ----
        if cfg.uniform_ratio < 1 and not uniformed and exit_layer is None and E == cfg.uniform_layer:
            keep_v = uniform_visual_keep(cfg.uniform_ratio).to(dev0) + img_s
            keep = torch.cat((torch.arange(img_s, device=dev0), keep_v, torch.arange(img_s + img_n, n0, device=dev0)))
            sel = torch.isin(cur_pos, keep)
            h, cur_pos, vis = h[:, sel.to(h.device)], cur_pos[sel], vis[sel]
            uniformed = True
        # ---- exit decision (uses the state BEFORE layer E, like Phase A) ----
        if exit_layer is None and cfg.exit != "none":
            if cfg.exit == "fixed":
                do_exit = (E == cfg.exit_layer)
            else:
                if cfg.exit == "lens":
                    lp = lens_logprobs(h[:, -1])
                    val = float((lp.exp() * (lp - prev_lp.to(lp.device))).sum()) if prev_lp is not None else float("inf")
                    prev_lp = lp
                else:
                    val = cf_last_res(layer, h, cur_pos[None], vis)
                run = run + 1 if val <= cfg.tau else 0
                do_exit = E >= cfg.e_min and run >= cfg.m
            if do_exit:
                sel = ~vis
                h, cur_pos, vis = h[:, sel.to(h.device)], cur_pos[sel], vis[sel]
                exit_layer = E
        seq_lens.append(h.shape[1])
        h = run_layer(layer, h, cur_pos[None], cache, cur_pos)
    logits = LM_HEAD(NORM(h[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float()
    nxt = int(logits.argmax(-1))
    _sync(); t1 = time.perf_counter()

    # ---- greedy decode with the (per-layer, possibly text-only) KV cache ----
    tokens = [nxt]
    for t in range(1, max_new_tokens):
        if stop_at_eos and nxt == EOS:
            break
        pos = torch.tensor([n0 + t - 1], device=dev0)
        h = EMBED(torch.tensor([[nxt]], device=_dev(EMBED))).to(dev0)
        for layer in LAYERS:
            h = run_layer(layer, h, pos[None], cache, pos)
        nxt = int(LM_HEAD(NORM(h[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float().argmax(-1))
        tokens.append(nxt)
    _sync(); t2 = time.perf_counter()
    text = tokenizer.decode([t for t in tokens if t != EOS], skip_special_tokens=True).strip()
    full = layer_flops(n0) * N_LAYERS
    return dict(text=text, tokens=tokens, exit_layer=exit_layer, seq_lens=seq_lens,
                rel_flops=sum(layer_flops(n) for n in seq_lens) / full, n0=n0,
                t_prefill=t1 - t0, t_decode=t2 - t1, t_total=t2 - t0)

def layer_flops(n):
    return 4 * n * D_MODEL**2 + 2 * n * n * D_MODEL + 2 * n * D_MODEL * D_FFN

def free_mem():
    gc.collect(); torch.cuda.empty_cache()
print("engine ready")
''')

md(r'''
## 5. Unit tests — the engine must reproduce the stock model
1. `baseline` prefill logits == stock `model(...)` logits (fp16 noise only).
2. `baseline` greedy text == stock `model.generate(use_cache=True)`.
3. `fixed_E` prefill logits == stock model run on text-only embeddings from layer E (checks the exit path: positions,
   cache shapes).
4. A decode step after exit runs without error and the cache has the expected per-layer lengths.
''')
code(r'''
demo = Image.open(urllib.request.urlopen("http://images.cocodataset.org/val2017/000000039769.jpg")).convert("RGB")
inputs = build_inputs(demo, "What is shown in this image? Answer in one sentence.")

with torch.no_grad():
    stock_logits = model(**inputs, use_cache=False).logits[0, -1].float()
    stock_gen = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
stock_text = processor.decode(stock_gen[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

emb = merged_embeddings(inputs)
n0 = emb.shape[1]; img_s, img_n = visual_span(inputs["input_ids"], n0)
print(f"merged seq len {n0}, image span [{img_s}, {img_s + img_n})")

res = engine_generate(inputs, ExitConfig("none"), 12)
with torch.no_grad():
    # recompute the engine's prefill logits for test 1 (engine_generate only keeps the argmax)
    cache = DynamicCache(); h = emb; pos = torch.arange(n0, device=emb.device)
    for layer in LAYERS:
        h = run_layer(layer, h, pos[None], cache, pos)
    eng_logits = LM_HEAD(NORM(h[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float()[0]
d1 = (eng_logits.to(stock_logits.device) - stock_logits).abs().max().item()
print(f"[1] max |logit diff| engine vs stock: {d1:.4f}")
print(f"[2] stock : {stock_text!r}\n    engine: {res['text']!r}")
assert d1 < 0.1, "engine prefill does not match the stock model"
stock_new = stock_gen[0, inputs["input_ids"].shape[1]:].tolist()
k = min(len(stock_new), len(res["tokens"]), 6)
assert res["tokens"][:k] == stock_new[:k], f"greedy tokens differ: {res['tokens'][:k]} vs {stock_new[:k]}"

# [3] fixed exit at E: reference = stock layers on the text-only sequence from E onward
E = 10
with torch.no_grad():
    cache = DynamicCache(); h = emb; pos = torch.arange(n0, device=emb.device)
    for layer in LAYERS[:E]:
        h = run_layer(layer, h, pos[None], cache, pos)
    keep = torch.cat((torch.arange(img_s, device=emb.device), torch.arange(img_s + img_n, n0, device=emb.device)))
    h = h[:, keep.to(h.device)]; pos = keep
    for layer in LAYERS[E:]:
        h = run_layer(layer, h, pos[None], cache, pos)
    ref_logits = LM_HEAD(NORM(h[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float()[0]
    # the engine's own fixed-exit path
    cfgE = ExitConfig("fixed", exit_layer=E)
    cache2 = DynamicCache(); h2 = emb; cur = torch.arange(n0, device=emb.device)
    vis = torch.zeros(n0, dtype=torch.bool, device=emb.device); vis[img_s:img_s + img_n] = True
    for i, layer in enumerate(LAYERS):
        if i == E:
            h2, cur = h2[:, (~vis).to(h2.device)], cur[~vis]
        h2 = run_layer(layer, h2, cur[None], cache2, cur)
    eng_logits_E = LM_HEAD(NORM(h2[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float()[0]
d3 = (eng_logits_E - ref_logits).abs().max().item()
print(f"[3] fixed-exit path vs reference: {d3:.4f}")
assert d3 < 1e-3
resE = engine_generate(inputs, cfgE, 8)
lens = [cache2.key_cache[i].shape[-2] if hasattr(cache2, "key_cache") else cache2.layers[i].keys.shape[-2] for i in (0, E - 1, E, 31)]
print(f"[4] exit@{E}: text={resE['text']!r} | seq_lens: layers<E {resE['seq_lens'][0]}, >=E {resE['seq_lens'][E]} | "
      f"cache lens (L0, L{E-1}, L{E}, L31) = {lens} | rel FLOPs {resE['rel_flops']:.3f}")
assert resE["seq_lens"][E] == n0 - img_n
print("\nALL UNIT TESTS PASSED")
free_mem()
''')

md(r'''
## 6. Exit thresholds from Phase A profiles (pooled across benchmarks, target fidelity 98%)
Point `PHASEA_DIR` at the folder holding `profile_*.jsonl` (e.g. an attached dataset). Reproduces the pooled
calibration of Phase A §8.5 and yields the `(tau, m, e_min)` the engine uses for `lens` and `cf`.
''')
code(r'''
PHASEA_DIR = next((p.parent for p in Path("/kaggle/input").rglob("profile_pope.jsonl")), Path("/kaggle/working/phaseA"))
T_POOL = 0.98
SIG = {"lens": ("lens_kl_prev", "low"), "cf": ("cf_last_res", "low")}

def _policy_exit(S, direction, tau, m, e_min):
    cond = (S <= tau) if direction == "low" else (S >= tau)
    cond = np.nan_to_num(cond, nan=False).astype(int)
    run = np.zeros_like(cond)
    for e in range(cond.shape[1]):
        run[:, e] = (run[:, e - 1] + 1) * cond[:, e] if e > 0 else cond[:, e]
    ok = run >= m; ok[:, :e_min] = False
    return np.concatenate([ok, np.ones((ok.shape[0], 1), bool)], 1).argmax(1)

def _flops_table(ns):
    ns = np.asarray(ns, float); E = np.arange(N_LAYERS + 1)[None]
    full, txt = layer_flops(ns)[:, None], layer_flops(ns - IMAGE_LEN)[:, None]
    return (E * full + (N_LAYERS - E) * txt) / (N_LAYERS * full)

profs = [json.loads(l) for p in sorted(PHASEA_DIR.glob("profile_*.jsonl")) for l in open(p)]
print(f"{len(profs)} Phase A profiles from {PHASEA_DIR}:", Counter(r["bench"] for r in profs))
pres = np.array([r["preserved"] for r in profs], bool); fl = _flops_table([r["n"] for r in profs])
CALIB = {}
for name, (key, d) in SIG.items():
    S = np.array([r[key][:N_LAYERS] for r in profs], float)
    finite = S[np.isfinite(S)]
    best = None
    for tau in np.unique(np.quantile(finite, np.linspace(0.02, 0.98, 25))):
        for m in (1, 2, 3):
            for e_min in (1, 2, 4, 8, 12):
                Ex = _policy_exit(S, d, tau, m, e_min); r_ = np.arange(len(Ex))
                f, c = pres[r_, Ex].mean(), fl[r_, Ex].mean()
                if f >= T_POOL and (best is None or c < best[0]):
                    best = (c, f, float(tau), m, e_min)
    c, f, tau, m, e_min = best
    CALIB[name] = dict(tau=tau, m=m, e_min=e_min)
    print(f"{name}: tau={tau:.4g} m={m} e_min={e_min}  -> pooled fidelity {f*100:.1f}% @ {c*100:.1f}% FLOPs (Phase A estimate)")
e_fixed = next(e for e in range(N_LAYERS + 1) if pres[:, e].mean() >= T_POOL)
print(f"fixed exit layer for the same pooled fidelity: E={e_fixed} @ {fl[:, e_fixed].mean()*100:.1f}% FLOPs")
json.dump({"calib": CALIB, "e_fixed": e_fixed}, open(OUT / "calib.json", "w"))
''')

md(r'''
## 7. Data (as in Phase A)
''')
code(r'''
%%bash
mkdir -p /kaggle/working/data && cd /kaggle/working/data
[ -d POPE ] || git clone --depth 1 https://github.com/AoiDragon/POPE.git
''')
code(r'''
N_PER_SPLIT, N_GQA = 300, 600
GQA_SUFFIX = "\nAnswer the question using a single word or phrase."
pope_dir = DATA / "POPE/output/coco"; coco_dir = DATA / "coco_val2014_subset"; coco_dir.mkdir(exist_ok=True)
random.seed(0); pope_items = []
for split in ["coco_pope_random.json", "coco_pope_popular.json", "coco_pope_adversarial.json"]:
    lines = [json.loads(l) for l in open(pope_dir / split) if l.strip()]
    images = sorted(set(l["image"] for l in lines)); random.shuffle(images)
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

GQA_Q_FILE = next(Path("/kaggle/input").rglob("testdev_balanced_questions.json"))
gqa_q = json.load(open(GQA_Q_FILE))
def find_image_dir(sample_image_id, root="/kaggle/input"):
    for dirpath, dirnames, filenames in os.walk(root):
        if f"{sample_image_id}.jpg" in filenames:
            return Path(dirpath)
        if len(filenames) > 1000:
            dirnames[:] = []
    raise FileNotFoundError("GQA images not found")
qids = list(gqa_q.keys()); random.Random(0).shuffle(qids)
GQA_IMG_DIR = find_image_dir(gqa_q[qids[0]]["imageId"])
gqa_items = [{"id": q, "image_path": str(GQA_IMG_DIR / f"{gqa_q[q]['imageId']}.jpg"), "question": gqa_q[q]["question"],
              "answer": gqa_q[q]["answer"], "structural": gqa_q[q].get("types", {}).get("structural")} for q in qids[:N_GQA]]
BENCH = {"pope": (pope_items, "", "yesno"), "gqa": (gqa_items, GQA_SUFFIX, "open")}
print("POPE", len(pope_items), "| GQA", len(gqa_items))
''')
code(r'''
# Optional: TextVQA + MME (same loaders as Phase A v2 §6b). Skip if not needed.
RUN_EXTRA_BENCH = False
if RUN_EXTRA_BENCH:
    from datasets import load_dataset
    SINGLE_WORD = "\nAnswer the question using a single word or phrase."
    def save_img(img, path):
        if not path.exists():
            img.convert("RGB").save(path, format="JPEG", quality=95)
    tv_dir = DATA / "textvqa"; tv_dir.mkdir(exist_ok=True)
    tv = load_dataset("lmms-lab/textvqa", split="validation").shuffle(seed=0).select(range(500))
    textvqa_items = []
    for ex in tv:
        path = tv_dir / f"{ex['question_id']}.jpg"; save_img(ex["image"], path)
        textvqa_items.append({"id": f"tv:{ex['question_id']}", "image_path": str(path), "question": ex["question"], "answer": list(ex["answers"])})
    mme_dir = DATA / "mme"; mme_dir.mkdir(exist_ok=True)
    try:
        mme = load_dataset("lmms-lab/MME", split="test")
    except Exception:
        mme = load_dataset("lmms-lab-encoder/MME", split="test")
    by_cat = defaultdict(list)
    for i, ex in enumerate(mme):
        by_cat[ex["category"]].append(i)
    rng = random.Random(0); mme_items = []
    for cat, idxs in sorted(by_cat.items()):
        for i in sorted(rng.sample(idxs, min(50, len(idxs)))):
            ex = mme[i]; path = mme_dir / f"{cat}_{ex['question_id'].replace('/', '_')}.jpg"; save_img(ex["image"], path)
            mme_items.append({"id": f"mme:{cat}:{ex['question_id']}", "image_path": str(path), "category": cat,
                              "question": ex["question"], "label": ex["answer"].strip().lower()})
    BENCH["textvqa"] = (textvqa_items, SINGLE_WORD, "open"); BENCH["mme"] = (mme_items, "", "yesno")
    print("TextVQA", len(textvqa_items), "| MME", len(mme_items))
''')
md(r'''
## 8. Latency micro-benchmark (batch 1, 20 POPE images, 8 new tokens, EOS ignored)
Compares the stock HF generate (SDPA + cache), the engine baseline, fixed exit, adaptive exit, and the Phase C variant.
If you also want FastV's eager-attention cost on this table, load a second model with `attn_implementation="eager"` on
the fork and time `model.generate(use_cache=False)` -- not done here to save memory.
''')
code(r'''
LAT_N, LAT_TOK = 20, 8
lat_imgs = [Image.open(it["image_path"]).convert("RGB") for it in pope_items[::len(pope_items) // LAT_N][:LAT_N]]
lat_q = "Describe the main object in this image in one sentence."
lat_cfgs = [ExitConfig("none"), ExitConfig("fixed", exit_layer=e_fixed), ExitConfig("fixed", exit_layer=16),
            ExitConfig("lens", **CALIB["lens"]), ExitConfig("cf", **CALIB["cf"]),
            ExitConfig("lens", **CALIB["lens"], uniform_ratio=0.5, uniform_layer=2)]

def time_stock(inputs):
    _sync(); t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=LAT_TOK, min_new_tokens=LAT_TOK, do_sample=False, use_cache=True)
    _sync(); return time.perf_counter() - t0

# warm-up
for img in lat_imgs[:2]:
    inp = build_inputs(img, lat_q); time_stock(inp); engine_generate(inp, lat_cfgs[0], LAT_TOK, stop_at_eos=False)

rows = defaultdict(list)
for img in lat_imgs:
    inp = build_inputs(img, lat_q)
    rows["stock_hf_generate"].append(dict(t_total=time_stock(inp), t_prefill=float("nan"), rel_flops=1.0, exit_layer=None))
    for cfg in lat_cfgs:
        r = engine_generate(inp, cfg, LAT_TOK, stop_at_eos=False)
        rows[cfg.name].append(r)
print(f"{'method':<34}{'prefill ms':>11}{'e2e ms':>9}{'FLOPs':>8}{'mean exit':>10}")
for name, rs in rows.items():
    pf = np.nanmean([r["t_prefill"] for r in rs]) * 1000; tot = np.mean([r["t_total"] for r in rs]) * 1000
    ex = [r["exit_layer"] for r in rs if r["exit_layer"] is not None]
    print(f"{name:<34}{pf:>11.0f}{tot:>9.0f}{np.mean([r['rel_flops'] for r in rs])*100:>7.1f}%{(np.mean(ex) if ex else float('nan')):>10.1f}")
json.dump({k: v for k, v in rows.items()}, open(OUT / "latency.json", "w"), default=str)
free_mem()
''')

md(r'''
## 9. Evaluation (resumable)
''')
code(r'''
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

def score(bench, kind, ans, it):
    if kind == "yesno":
        p = parse_yes_no(ans); return {"pred": p, "correct": p == it["label"]}
    p = norm_answer(ans)
    if isinstance(it["answer"], list):                      # VQA-style soft accuracy (TextVQA)
        gts = [norm_answer(a) for a in it["answer"]]
        return {"pred": p, "correct": min(1.0, sum(g == p for g in gts) / 3.0)}
    return {"pred": p, "correct": p == norm_answer(it["answer"])}

def run_eval(bench, cfg, max_items=None):
    items, suffix, kind = BENCH[bench]
    items = items[:max_items] if max_items else items
    path = OUT / f"{bench}__{cfg.name}.jsonl"
    # MME ids repeat (two questions per image): key cached records by (id, k-th occurrence), in item order
    def occ_keys(ids):
        seen = Counter(); out = []
        for i in ids:
            out.append((i, seen[i])); seen[i] += 1
        return out
    done = {}
    if path.exists():
        recs = [json.loads(l) for l in open(path)]
        done = dict(zip(occ_keys([r["id"] for r in recs]), recs))
    keys = occ_keys([it["id"] for it in items])
    todo = [(k, it) for k, it in zip(keys, items) if k not in done]
    print(f"[{bench}/{cfg.name}] {len(done)} cached, {len(todo)} to run")
    mnt = 3 if kind == "yesno" else 10
    with open(path, "a") as f:
        for i, (k, it) in enumerate(todo):
            inp = build_inputs(Image.open(it["image_path"]).convert("RGB"), it["question"], suffix)
            r = engine_generate(inp, cfg, mnt)
            rec = {"id": it["id"], "answer": r["text"], "exit_layer": r["exit_layer"], "rel_flops": round(r["rel_flops"], 4),
                   "t_prefill": round(r["t_prefill"], 4), "t_total": round(r["t_total"], 4), "n0": r["n0"], **score(bench, kind, r["text"], it)}
            f.write(json.dumps(rec) + "\n"); f.flush(); done[k] = rec
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(todo)}")
    free_mem()
    recs = [done[k] for k in keys]
    if todo:   # keep the file in item order (a restored subset run may have been appended out of order)
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    return recs

METHODS = [ExitConfig("none"),
           ExitConfig("lens", **CALIB["lens"]),
           ExitConfig("fixed", exit_layer=e_fixed),
           ExitConfig("cf", **CALIB["cf"]),
           ExitConfig("fixed", exit_layer=16), ExitConfig("fixed", exit_layer=20), ExitConfig("fixed", exit_layer=24),
           ExitConfig("lens", **CALIB["lens"], uniform_ratio=0.5, uniform_layer=2),       # Phase C
           ExitConfig("none", uniform_ratio=0.5, uniform_layer=2)]
RUNS = {b: {cfg.name: run_eval(b, cfg) for cfg in METHODS} for b in BENCH}
os.system(f"cd {WORK} && zip -rq phaseB_results.zip phaseB")
print("zipped ->", WORK / "phaseB_results.zip")
''')
code(r'''
def mcnemar(a, b):
    b01 = sum(x and not y for x, y in zip(a, b)); b10 = sum(y and not x for x, y in zip(a, b)); n = b01 + b10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2 ** n)
    return b01, b10, p

for b, runs in RUNS.items():
    base = runs["baseline"]
    base_c = [bool(r["correct"]) if not isinstance(r["correct"], float) else r["correct"] >= 1 for r in base]
    print(f"\n==== {b.upper()} (n={len(base)}) ====")
    print(f"{'method':<36}{'acc':>7}{'agree':>8}{'FLOPs':>8}{'exit μ':>8}{'prefill ms':>11}{'e2e ms':>8}{'vs base p':>10}")
    for name, recs in runs.items():
        acc = np.mean([r["correct"] for r in recs]); agree = np.mean([x["pred"] == y["pred"] for x, y in zip(recs, base)])   # by position (MME ids repeat)
        ex = [r["exit_layer"] for r in recs if r["exit_layer"] is not None]
        c = [bool(r["correct"]) if not isinstance(r["correct"], float) else r["correct"] >= 1 for r in recs]
        _, _, p = mcnemar(c, base_c)
        print(f"{name:<36}{acc*100:>6.1f}%{agree*100:>7.1f}%{np.mean([r['rel_flops'] for r in recs])*100:>7.1f}%"
              f"{(np.mean(ex) if ex else float('nan')):>8.1f}{np.mean([r['t_prefill'] for r in recs])*1000:>11.0f}"
              f"{np.mean([r['t_total'] for r in recs])*1000:>8.0f}{p:>10.3f}")
''')

md(r'''
## What to send back
`phaseB_results.zip` (or: the §5 unit-test output, §6 thresholds, §7 latency table, §8 tables). The key comparisons:
`lens_*` vs `fixed_E{e_fixed}` at the same pooled fidelity target (accuracy, agreement, FLOPs, **measured** latency), and
whether `*_uni0.5@L2` keeps agreement while cutting prefill further.
''')

nb = {"cells": [({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)} if t == "markdown" else
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open("../../notebooks/phaseB-exit-engine.ipynb", "w"), indent=1)
print(len(cells), "cells")
