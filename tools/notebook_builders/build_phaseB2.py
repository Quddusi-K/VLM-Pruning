import json, re

old = open("build_phaseB.py").read()
def grab(start_marker, end_marker):
    i = old.index(start_marker); j = old.index(end_marker, i + 1)
    return old[i:j]

cells = []
def md(s): cells.append(("markdown", s.strip("\n")))
def code(s): cells.append(("code", s.strip("\n")))

md(r'''
# Phase B v2 — engine with cache-based probes, FastV-in-engine, and a second model (LLaVA-NeXT-7B)

Changes vs Phase B:
- **Counterfactual probe from the KV cache.** After layer E runs, the probe reads layer E's cached K/V (post-RoPE) and
  recomputes only the last token's query → one row of attention. Same quantity as before (the Phase A thresholds
  transfer), decision applied before layer E+1. No more full-sequence K/V recomputation.
- **FastV inside the engine.** The same one-row attention gives FastV's criterion (last-token attention over image tokens
  at layer K−1), so FastV K/R runs on SDPA + KV cache here, for a like-for-like latency comparison.
- **Model switch.** `MODEL_TAG = "llava15"` or `"llavanext"` (`llava-hf/llava-v1.6-vicuna-7b-hf`, up to ~2.9k visual
  tokens). Results go to `/kaggle/working/phaseB2/<MODEL_TAG>/`. LLaVA-NeXT is run with the **same τ as LLaVA-1.5**
  (no recalibration) on smaller subsets.

Run order: §1–§3, **§4 unit tests**, §5 data (+ restore of previous LLaVA-1.5 results), §6 latency, §7 evaluation.
''')

md("## 1. Environment + install (stock transformers 4.49). Restart the kernel after the install if transformers was already imported.")
code(r'''
import subprocess, sys, torch
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
subprocess.run(f"{sys.executable} -m pip install -q transformers==4.49.0 accelerate sentencepiece pillow datasets", shell=True, check=True)
print(subprocess.run([sys.executable, "-c", "import transformers; print('transformers', transformers.__version__)"], capture_output=True, text=True).stdout)
''')

md("## 2. Model (fp16, SDPA, split across GPUs)")
code(r'''
MODEL_TAG = "llava15"          # "llava15" | "llavanext"
MODELS = {"llava15": "llava-hf/llava-1.5-7b-hf", "llavanext": "llava-hf/llava-v1.6-vicuna-7b-hf"}

import os
os.environ["USE_TF"] = "0"
import gc, json, math, random, re, shutil, string, time, urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
print("transformers", transformers.__version__)

MODEL_ID = MODELS[MODEL_TAG]
WORK = Path("/kaggle/working"); DATA, OUT = WORK / "data", WORK / "phaseB2" / MODEL_TAG
for p in (DATA, OUT):
    p.mkdir(parents=True, exist_ok=True)
N_LAYERS, D_MODEL, D_FFN = 32, 4096, 11008

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
if "model" not in globals() or globals().get("_loaded_id") != MODEL_ID:
    if "model" in globals():
        del model; gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float16, attn_implementation="sdpa", device_map="auto")
    model.eval(); _loaded_id = MODEL_ID

_lm = model.language_model
if hasattr(_lm, "model") and hasattr(_lm, "lm_head"):
    CORE, LM_HEAD = _lm.model, _lm.lm_head
else:
    CORE, LM_HEAD = _lm, model.lm_head
LAYERS, NORM, EMBED = CORE.layers, CORE.norm, CORE.embed_tokens
ROTARY = getattr(CORE, "rotary_emb", None)
IMG_TOK = getattr(model.config, "image_token_index", None) or model.config.image_token_id
EOS = tokenizer.eos_token_id
attn_impl = getattr(CORE.config, "_attn_implementation", "?")
print(f"{MODEL_ID}: layers={len(LAYERS)} attn={attn_impl} devices={sorted({str(p.device) for p in model.parameters()})}")
assert "sdpa" in attn_impl.lower()
''')

md(r'''
## 3. Engine v2
''')
code(r'''
def _dev(m):
    return next(m.parameters()).device

class _Captured(Exception):
    pass
_CAP = {}
def _capture_hook(module, args, kwargs):
    emb = kwargs.get("inputs_embeds")
    _CAP["emb"] = emb if emb is not None else (args[0] if args else None)
    raise _Captured()

@torch.no_grad()
def merged_embeddings(inputs):
    """Text+image embeddings as the library's own LLaVA/LLaVA-NeXT forward builds them (captured by a pre-hook)."""
    h = model.language_model.register_forward_pre_hook(_capture_hook, with_kwargs=True)
    try:
        model(**inputs, use_cache=False)
    except _Captured:
        pass
    finally:
        h.remove()
    emb = _CAP.pop("emb")
    assert emb is not None and emb.dim() == 3 and emb.shape[0] == 1
    return emb

def visual_span(input_ids, n_embed):
    pos = (input_ids[0] == IMG_TOK).nonzero().flatten()
    assert pos.numel() >= 1, "no image token"
    if pos.numel() > 1:
        assert int(pos[-1] - pos[0]) == pos.numel() - 1, "image tokens not contiguous"
        assert n_embed == input_ids.shape[1]
        return int(pos[0]), int(pos.numel())
    return int(pos[0]), n_embed - input_ids.shape[1] + 1

def build_inputs(image, question, suffix=""):
    prompt = f"USER: <image>\n{question}{suffix}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    return inputs

def layer_flops(n):
    return 4 * n * D_MODEL**2 + 2 * n * n * D_MODEL + 2 * n * D_MODEL * D_FFN

def lens_logprobs(h_last):
    z = NORM(h_last.to(_dev(NORM)))
    return LM_HEAD(z.to(_dev(LM_HEAD))).float().log_softmax(-1)[0]

def rope_for(layer, h, pos_ids):
    if ROTARY is None:
        return None
    dev = _dev(layer)
    return ROTARY(h.to(dev), pos_ids.to(dev))

def run_layer(layer, h, pos_ids, cache, cache_pos):
    dev = _dev(layer)
    kw = dict(attention_mask=None, position_ids=pos_ids.to(dev), past_key_value=cache,
              output_attentions=False, use_cache=True, cache_position=cache_pos.to(dev))
    pe = rope_for(layer, h, pos_ids)
    if pe is not None:
        kw["position_embeddings"] = pe
    out = layer(h.to(dev), **kw)
    return out[0] if isinstance(out, (tuple, list)) else out

def cache_kv(cache, E):
    """Layer E's cached keys/values (1, Hkv, n, hd), already RoPE-rotated. Works for 4.4x–4.5x DynamicCache layouts."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[E], cache.value_cache[E]
    return cache.layers[E].keys, cache.layers[E].values

@torch.no_grad()
def probe_row(layer, E, h_in, pos_ids, cache, vis_mask):
    """One row of attention for the LAST token of layer E, from the cache.
    Returns (p_mean over heads: (n,), cf value: relative change of the attention update without visual keys)."""
    attn, dev = layer.self_attn, _dev(layer)
    k, v = cache_kv(cache, E)
    k, v = k.to(dev), v.to(dev)
    H = getattr(attn, "num_heads", None) or attn.config.num_attention_heads
    Hkv = k.shape[1]; hd = k.shape[-1]
    x = layer.input_layernorm(h_in[:, -1:].to(dev))
    q = attn.q_proj(x).view(1, 1, H, hd).transpose(1, 2)
    pe = rope_for(layer, h_in[:, -1:], pos_ids[:, -1:])
    cos, sin = pe if pe is not None else attn.rotary_emb(v, pos_ids[:, -1:].to(dev))
    q = q * cos.to(dev).unsqueeze(1) + rotate_half(q) * sin.to(dev).unsqueeze(1)
    k, v = repeat_kv(k, H // Hkv), repeat_kv(v, H // Hkv)
    s = (q.float() @ k.float().transpose(2, 3)) / math.sqrt(hd)                       # (1, H, 1, n)
    p = s.softmax(-1)
    vm = vis_mask.to(dev)
    p_txt = s.masked_fill(vm[None, None, None, :], float("-inf")).softmax(-1)
    o = (((p - p_txt) @ v.float()).transpose(1, 2).reshape(1, 1, H * hd)).to(x.dtype)
    delta = attn.o_proj(o).float()[0, 0]
    cf = (delta.norm() / h_in[0, -1].to(dev).float().norm().clamp_min(1e-6)).item()
    return p.mean(1)[0, 0], cf


class ExitConfig:
    """exit: 'none' | 'fixed' | 'lens' | 'cf'.
       lens: before layer E, exit when KL(lens_E || lens_{E-1}) <= tau for m consecutive layers, E >= e_min.
       cf:   after layer E (cache probe), same rule → exit before layer E+1.
       prune: 'none' | 'fastv' (after layer fastv_k-1, keep the top (1-fastv_r) visual tokens by last-token attention)."""
    def __init__(self, exit="none", exit_layer=None, tau=None, m=1, e_min=1, prune="none", fastv_k=2, fastv_r=0.5, name=None):
        self.exit, self.exit_layer, self.tau, self.m, self.e_min = exit, exit_layer, tau, m, e_min
        self.prune, self.fastv_k, self.fastv_r = prune, fastv_k, fastv_r
        self.name = name or self.auto_name()
    def auto_name(self):
        parts = []
        if self.prune == "fastv":
            parts.append(f"fastv_K{self.fastv_k}_R{self.fastv_r}")
        if self.exit == "fixed":
            parts.append(f"fixed_E{self.exit_layer}")
        elif self.exit in ("lens", "cf"):
            parts.append(f"{self.exit}_t{self.tau:.3g}_m{self.m}_e{self.e_min}")
        return "+".join(parts) or "baseline"

def _sync():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)

@torch.no_grad()
def engine_generate(inputs, cfg, max_new_tokens, stop_at_eos=True):
    _sync(); t0 = time.perf_counter()
    emb = merged_embeddings(inputs)
    n0 = emb.shape[1]
    img_s, img_n = visual_span(inputs["input_ids"], n0)
    dev0 = emb.device
    cur_pos = torch.arange(n0, device=dev0)
    vis = torch.zeros(n0, dtype=torch.bool, device=dev0); vis[img_s:img_s + img_n] = True
    cache = DynamicCache()
    h = emb
    exit_layer, pruned, prev_lp, run = None, False, None, 0
    seq_lens = []
    L = len(LAYERS)
    for E, layer in enumerate(LAYERS):
        # ---- pre-layer decisions: fixed / lens ----
        if exit_layer is None and cfg.exit in ("fixed", "lens"):
            if cfg.exit == "fixed":
                do_exit = (E == cfg.exit_layer)
            else:
                lp = lens_logprobs(h[:, -1])
                val = float((lp.exp() * (lp - prev_lp.to(lp.device))).sum()) if prev_lp is not None else float("inf")
                prev_lp = lp
                run = run + 1 if val <= cfg.tau else 0
                do_exit = E >= cfg.e_min and run >= cfg.m
            if do_exit:
                sel = ~vis; h, cur_pos, vis = h[:, sel.to(h.device)], cur_pos[sel], vis[sel]; exit_layer = E
        seq_lens.append(h.shape[1])
        h_in = h
        h = run_layer(layer, h, cur_pos[None], cache, cur_pos)
        # ---- post-layer decisions from this layer's cache: cf exit / FastV prune ----
        need_cf = cfg.exit == "cf" and exit_layer is None and E + 1 < L
        need_fv = cfg.prune == "fastv" and not pruned and E == cfg.fastv_k - 1 and exit_layer is None
        if (need_cf or need_fv) and vis.any():
            p_mean, cf_val = probe_row(layer, E, h_in, cur_pos[None], cache, vis)
            if need_cf:
                run = run + 1 if cf_val <= cfg.tau else 0
                if E + 1 >= cfg.e_min and run >= cfg.m:
                    sel = ~vis; h, cur_pos, vis = h[:, sel.to(h.device)], cur_pos[sel], vis[sel]; exit_layer = E + 1
            if need_fv and exit_layer is None:
                vis_idx = vis.nonzero().flatten()
                keep_n = round(vis_idx.numel() * (1 - cfg.fastv_r))
                scores = p_mean.to(dev0)[vis_idx]
                keep_vis = vis_idx[scores.topk(keep_n).indices]
                sel = ~vis; sel[keep_vis] = True
                h, cur_pos, vis = h[:, sel.to(h.device)], cur_pos[sel], vis[sel]; pruned = True
    logits = LM_HEAD(NORM(h[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float()
    nxt = int(logits.argmax(-1))
    _sync(); t1 = time.perf_counter()
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
    return dict(text=text, tokens=tokens, exit_layer=exit_layer, seq_lens=seq_lens, n0=n0, n_vis=img_n,
                rel_flops=sum(layer_flops(n) for n in seq_lens) / (layer_flops(n0) * L),
                t_prefill=t1 - t0, t_decode=t2 - t1, t_total=t2 - t0)

def free_mem():
    gc.collect(); torch.cuda.empty_cache()
print("engine v2 ready")
''')

md(r'''
## 4. Unit tests
(1) engine == stock logits; (2) engine == stock greedy text; (3) the cache probe reproduces the direct computation of
FastV's last-token attention at layer 1; (4) FastV mode removes the right number of tokens and a cf exit runs.
''')
code(r'''
demo = Image.open(urllib.request.urlopen("http://images.cocodataset.org/val2017/000000039769.jpg")).convert("RGB")
inputs = build_inputs(demo, "What is shown in this image? Answer in one sentence.")
with torch.no_grad():
    stock_logits = model(**inputs, use_cache=False).logits[0, -1].float()
    stock_gen = model.generate(**inputs, max_new_tokens=12, do_sample=False, use_cache=True)
stock_text = processor.decode(stock_gen[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
emb = merged_embeddings(inputs); n0 = emb.shape[1]; img_s, img_n = visual_span(inputs["input_ids"], n0)
print(f"merged seq len {n0}, image span [{img_s}, {img_s + img_n}) -> {img_n} visual tokens")

with torch.no_grad():
    cache = DynamicCache(); h = emb; pos = torch.arange(n0, device=emb.device)
    h_in_1 = None
    for i, layer in enumerate(LAYERS):
        if i == 1: h_in_1 = h
        h = run_layer(layer, h, pos[None], cache, pos)
    eng_logits = LM_HEAD(NORM(h[:, -1].to(_dev(NORM))).to(_dev(LM_HEAD))).float()[0]
d1 = (eng_logits.to(stock_logits.device) - stock_logits).abs().max().item()
res = engine_generate(inputs, ExitConfig("none"), 12)
stock_new = stock_gen[0, inputs["input_ids"].shape[1]:].tolist(); k = min(len(stock_new), len(res["tokens"]), 6)
print(f"[1] max |logit diff| engine vs stock: {d1:.4f}")
print(f"[2] stock : {stock_text!r}\n    engine: {res['text']!r}")
assert d1 < 0.1 and res["tokens"][:k] == stock_new[:k]

# [3] probe vs direct attention at layer 1 (eager recomputation of the full last-row attention)
with torch.no_grad():
    layer = LAYERS[1]; attn = layer.self_attn; dev = _dev(layer)
    vis = torch.zeros(n0, dtype=torch.bool, device=emb.device); vis[img_s:img_s + img_n] = True
    p_probe, cf_probe = probe_row(layer, 1, h_in_1, pos[None], cache, vis)
    x = layer.input_layernorm(h_in_1.to(dev)); n = x.shape[1]
    H = getattr(attn, "num_heads", None) or attn.config.num_attention_heads; hd = attn.head_dim
    Hkv = getattr(attn, "num_key_value_heads", None) or attn.config.num_key_value_heads
    q = attn.q_proj(x[:, -1:]).view(1, 1, H, hd).transpose(1, 2); kk = attn.k_proj(x).view(1, n, Hkv, hd).transpose(1, 2)
    pe = rope_for(layer, h_in_1, pos[None]); cos, sin = pe if pe is not None else attn.rotary_emb(kk, pos[None].to(dev))
    cos, sin = cos.to(dev), sin.to(dev)
    q = q * cos[:, -1:].unsqueeze(1) + rotate_half(q) * sin[:, -1:].unsqueeze(1)
    kk = kk * cos.unsqueeze(1) + rotate_half(kk) * sin.unsqueeze(1); kk = repeat_kv(kk, H // Hkv)
    p_direct = ((q.float() @ kk.float().transpose(2, 3)) / math.sqrt(hd)).softmax(-1).mean(1)[0, 0]
d3 = (p_probe - p_direct).abs().max().item()
print(f"[3] probe attention row vs direct: max diff {d3:.2e} | cf value at layer 1: {cf_probe:.4f}")
assert d3 < 1e-3

r_fv = engine_generate(inputs, ExitConfig(prune="fastv", fastv_k=2, fastv_r=0.5), 12)
r_cf = engine_generate(inputs, ExitConfig("cf", tau=0.05, m=2, e_min=12), 12)
print(f"[4] fastv K2 R0.5: {r_fv['text']!r} | seq len after prune {r_fv['seq_lens'][2]} (expected {n0 - round(img_n*0.5)}) | FLOPs {r_fv['rel_flops']:.3f}")
print(f"    cf exit: {r_cf['text']!r} | exit layer {r_cf['exit_layer']} | FLOPs {r_cf['rel_flops']:.3f}")
assert r_fv["seq_lens"][2] == n0 - round(img_n * 0.5) and r_fv["seq_lens"][1] == n0
print("\nALL UNIT TESTS PASSED")
free_mem()
''')

md(r'''
## 5. Data (same sampling as before; `N_SUB` caps each benchmark — use the full sets for LLaVA-1.5, 300 each for LLaVA-NeXT)
Previous LLaVA-1.5 results are restored from an attached `phaseB_results.zip` dataset so baseline / fixed / lens runs are reused.
''')
code(r'''
N_SUB = {"llava15": None, "llavanext": 300}[MODEL_TAG]
''')
code(grab("code(r'''\n%%bash\nmkdir -p /kaggle/working/data && cd /kaggle/working/data", "code(r'''\nN_PER_SPLIT, N_GQA = 300, 600")[len("code(r'''\n"):-len("''')\n")])
code(grab("code(r'''\nN_PER_SPLIT, N_GQA = 300, 600", "code(r'''\n# Optional: TextVQA + MME")[len("code(r'''\n"):-len("''')\n")].replace(
    'BENCH = {"pope": (pope_items, "", "yesno"), "gqa": (gqa_items, GQA_SUFFIX, "open")}\nprint("POPE", len(pope_items), "| GQA", len(gqa_items))',
    'BENCH = {"pope": (pope_items, "", "yesno"), "gqa": (gqa_items, GQA_SUFFIX, "open")}'))
code(grab("code(r'''\n# Optional: TextVQA + MME", "md(r'''\n## 8. Latency")[len("code(r'''\n"):-len("''')\n")].replace(
    "RUN_EXTRA_BENCH = False", "RUN_EXTRA_BENCH = True").replace(
    '    print("TextVQA", len(textvqa_items), "| MME", len(mme_items))',
    '''
if N_SUB:   # strided subset so POPE splits / MME categories stay mixed; ids are a subset of the full LLaVA-1.5 runs
    BENCH = {b: (items[::max(1, len(items) // N_SUB)][:N_SUB], suf, kind) for b, (items, suf, kind) in BENCH.items()}
for b, (items, _, _) in BENCH.items():
    print(b, len(items))
# restore previous results for this model (attach phaseB_results.zip / phaseB2 outputs as a dataset)
restored = 0
for p in Path("/kaggle/input").rglob("*__*.jsonl"):
    if MODEL_TAG == "llava15" and ("phaseB/" in str(p) or f"phaseB2/{MODEL_TAG}" in str(p)) or (MODEL_TAG != "llava15" and f"phaseB2/{MODEL_TAG}" in str(p)):
        if "cf_" in p.name or "uni" in p.name:      # old expensive-probe cf runs and uniform ablations: do not reuse
            continue
        if not (OUT / p.name).exists():
            shutil.copy(p, OUT / p.name); restored += 1
print("restored result files:", restored)'''))

md("## 6. Latency (20 POPE images, 8 new tokens, EOS ignored)")
code(r'''
LAT_N, LAT_TOK = 20, 8
pope_all, _, _ = BENCH["pope"]
lat_imgs = [Image.open(it["image_path"]).convert("RGB") for it in pope_all[::max(1, len(pope_all) // LAT_N)][:LAT_N]]
lat_q = "Describe the main object in this image in one sentence."
lat_cfgs = [ExitConfig("none"), ExitConfig("fixed", exit_layer=16), ExitConfig("fixed", exit_layer=24),
            ExitConfig("lens", tau=0.05), ExitConfig("cf", tau=0.05, m=2, e_min=12),
            ExitConfig(prune="fastv", fastv_k=2, fastv_r=0.5), ExitConfig(prune="fastv", fastv_k=2, fastv_r=0.75),
            ExitConfig("lens", tau=0.05, prune="fastv", fastv_k=2, fastv_r=0.5)]
def time_stock(inputs):
    _sync(); t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=LAT_TOK, min_new_tokens=LAT_TOK, do_sample=False, use_cache=True)
    _sync(); return time.perf_counter() - t0
for img in lat_imgs[:2]:
    inp = build_inputs(img, lat_q); time_stock(inp); engine_generate(inp, lat_cfgs[0], LAT_TOK, stop_at_eos=False)
rows = defaultdict(list)
for img in lat_imgs:
    inp = build_inputs(img, lat_q)
    rows["stock_hf_generate"].append(dict(t_total=time_stock(inp), t_prefill=float("nan"), rel_flops=1.0, exit_layer=None))
    for cfg in lat_cfgs:
        rows[cfg.name].append(engine_generate(inp, cfg, LAT_TOK, stop_at_eos=False))
print(f"{'method':<40}{'prefill ms':>11}{'e2e ms':>9}{'FLOPs':>8}{'mean exit':>10}")
for name, rs in rows.items():
    pf = np.nanmean([r["t_prefill"] for r in rs]) * 1000; tot = np.mean([r["t_total"] for r in rs]) * 1000
    ex = [r["exit_layer"] for r in rs if r["exit_layer"] is not None]
    print(f"{name:<40}{pf:>11.0f}{tot:>9.0f}{np.mean([r['rel_flops'] for r in rs])*100:>7.1f}%{(np.mean(ex) if ex else float('nan')):>10.1f}")
json.dump(rows, open(OUT / "latency.json", "w"), default=str)
free_mem()
''')

md("## 7. Evaluation (resumable; results reused by file name)")
code(grab("code(r'''\ndef parse_yes_no(text):", "code(r'''\ndef mcnemar(a, b):")[len("code(r'''\n"):-len("''')\n")]
     .replace('''METHODS = [ExitConfig("none"),
           ExitConfig("lens", **CALIB["lens"]),
           ExitConfig("fixed", exit_layer=e_fixed),
           ExitConfig("cf", **CALIB["cf"]),
           ExitConfig("fixed", exit_layer=16), ExitConfig("fixed", exit_layer=20), ExitConfig("fixed", exit_layer=24),
           ExitConfig("lens", **CALIB["lens"], uniform_ratio=0.5, uniform_layer=2),       # Phase C
           ExitConfig("none", uniform_ratio=0.5, uniform_layer=2)]''',
'''if MODEL_TAG == "llava15":
    METHODS = [ExitConfig("none"),
               ExitConfig(prune="fastv", fastv_k=2, fastv_r=0.5), ExitConfig(prune="fastv", fastv_k=2, fastv_r=0.75),
               ExitConfig(prune="fastv", fastv_k=3, fastv_r=0.5),
               ExitConfig("cf", tau=0.05, m=2, e_min=12), ExitConfig("cf", tau=0.1, m=2, e_min=12),
               ExitConfig("lens", tau=0.05), ExitConfig("lens", tau=0.05, prune="fastv", fastv_k=2, fastv_r=0.5),
               ExitConfig("fixed", exit_layer=16), ExitConfig("fixed", exit_layer=24)]
else:   # LLaVA-NeXT: same tau as LLaVA-1.5, no recalibration
    METHODS = [ExitConfig("none"), ExitConfig("lens", tau=0.05), ExitConfig("lens", tau=0.1),
               ExitConfig("cf", tau=0.05, m=2, e_min=12),
               ExitConfig("fixed", exit_layer=16), ExitConfig("fixed", exit_layer=24),
               ExitConfig(prune="fastv", fastv_k=2, fastv_r=0.5)]''')
     .replace('rec = {"id": it["id"], "answer": r["text"], "exit_layer": r["exit_layer"], "rel_flops": round(r["rel_flops"], 4),',
              'rec = {"id": it["id"], "answer": r["text"], "exit_layer": r["exit_layer"], "rel_flops": round(r["rel_flops"], 4), "n_vis": r["n_vis"],')
     .replace('os.system(f"cd {WORK} && zip -rq phaseB_results.zip phaseB")\nprint("zipped ->", WORK / "phaseB_results.zip")',
              'os.system(f"cd {WORK} && zip -rq phaseB2_{MODEL_TAG}.zip phaseB2/{MODEL_TAG}")\nprint("zipped ->", WORK / f"phaseB2_{MODEL_TAG}.zip")'))
code(grab("code(r'''\ndef mcnemar(a, b):", "md(r'''\n## What to send back")[len("code(r'''\n"):-len("''')\n\n")]
     .replace('base = runs["baseline"]; base_pred = {r["id"]: r["pred"] for r in base}',
              'base = runs["baseline"]')
     .replace('agree = np.mean([base_pred[r["id"]] == r["pred"] for r in recs])',
              'agree = np.mean([x["pred"] == y["pred"] for x, y in zip(recs, base)])   # by position (MME ids repeat)'))

md(r'''
## What to send back
`phaseB2_llava15.zip` and `phaseB2_llavanext.zip` (each has `latency.json` + per-method JSONL). From the printed
output: the §4 unit-test lines, and the §6 latency tables for both models.
''')

nb = {"cells": [({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)} if t == "markdown" else
                 {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.splitlines(keepends=True)})
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
json.dump(nb, open("../../notebooks/phaseB2-engine-v2.ipynb", "w"), indent=1)
print(len(cells), "cells")
