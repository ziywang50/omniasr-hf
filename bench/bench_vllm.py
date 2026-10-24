"""Offline throughput benchmark for OmniASR LLM 300M in vLLM.

  --impl native        vLLM's native implementation (PR #36864 branch)
  --impl transformers  Transformers backend + this repo's remote code (upstream vLLM, unchanged)
  --eager              disable CUDA graphs / torch.compile

Same workload for both: the 73 clips in bench_audio repeated 5x (365 requests),
greedy decoding, prefix caching and the multimodal processor cache disabled,
2 warmup passes, median of 5 timed runs. The last run's first 73 outputs are
compared with the native implementation's reference outputs.
"""
import argparse
import json
import re
import statistics
import time
from collections import Counter

import soundfile as sf
import torch
import vllm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

AUDIO_DIR = "/root/autodl-tmp/bench_audio"
NATIVE_REF = "/root/autodl-tmp/vllm_out.json"   # native implementation's outputs (read only)
HF_DIR = "/root/autodl-tmp/omniasr-hf"
REPEAT, WARMUP, RUNS = 5, 2, 5


def words(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(meta, hyps):
    err = sum(edit_distance(words(m["ref"]), words(h)) for m, h in zip(meta, hyps))
    return err / sum(len(words(m["ref"])) for m in meta)


def make_prompt(impl, audio):
    lang = {"language": "eng_Latn"}
    if impl == "native":
        # [PAD, BOS]: PAD is replaced by audio + lid + lang embeddings (same as get_generation_prompt)
        return TokensPrompt(prompt_token_ids=[1, 0], multi_modal_data={"audio": (audio, 16000)},
                            mm_processor_kwargs=lang)
    # Transformers backend: the HF processor expands "<pad>" into the placeholders; "<s>" is BOS
    return {"prompt": "<pad><s>", "multi_modal_data": {"audio": (audio, 16000)},
            "mm_processor_kwargs": lang}


def build_llm(impl, eager):
    common = dict(attention_backend="TRITON_ATTN", dtype="bfloat16", enforce_eager=eager,
                  gpu_memory_utilization=0.75, enable_prefix_caching=False, mm_processor_cache_gb=0)
    if impl == "native":
        return LLM(model="ziywang50/omniASR-LLM-300M", **common)
    return LLM(model=HF_DIR, model_impl="transformers", trust_remote_code=True,
               limit_mm_per_prompt={"audio": 1}, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["native", "transformers"], required=True)
    ap.add_argument("--eager", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(f"{AUDIO_DIR}/manifest.json"))
    audios = [sf.read(m["path"], dtype="float32")[0] for m in meta]
    inp = audios * REPEAT
    total_dur = sum(len(a) / 16000 for a in inp)

    llm = build_llm(args.impl, args.eager)
    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    prompts = [make_prompt(args.impl, a) for a in inp]

    print(f"CONFIG impl={args.impl} vllm={vllm.__version__} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name()}")
    try:
        sc = llm.llm_engine.vllm_config.scheduler_config
        print(f"CONFIG max_num_seqs={sc.max_num_seqs} max_num_batched_tokens={sc.max_num_batched_tokens}")
    except Exception as e:
        print(f"CONFIG scheduler config unavailable: {e!r} (see full log)")

    for _ in range(WARMUP):
        llm.generate(prompts[:len(meta)], sp, use_tqdm=False)

    times = []
    for run in range(RUNS):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        times.append(time.perf_counter() - t0)
        print(f"  run {run}: {times[-1]:.1f}s  RTFx={total_dur / times[-1]:.1f}", flush=True)
    assert len(outs) == len(inp), f"expected {len(inp)} outputs, got {len(outs)}"

    reasons = Counter(o.outputs[0].finish_reason for o in outs)
    print(f"FINISH {dict(reasons)}")
    for i, o in enumerate(outs[:len(meta)]):
        if o.outputs[0].finish_reason == "length":
            print(f"TRUNC [{i:03d}] {len(o.outputs[0].token_ids)} tokens, "
                  f"{meta[i]['dur']:.1f}s audio: {o.outputs[0].text[:300]}")
    assert reasons.get("length", 0) == 0, f"{reasons['length']}/{len(outs)} outputs hit max_tokens=1024"

    med = statistics.median(times)
    print(f"VLLM impl={args.impl} eager={args.eager} median={med:.2f}s RTFx={total_dur / med:.1f} "
          f"runs={[round(t, 2) for t in times]}")

    texts = [o.outputs[0].text for o in outs[:len(meta)]]
    out_path = f"/root/autodl-tmp/bench_out_{args.impl}{'_eager' if args.eager else ''}.json"
    json.dump(texts, open(out_path, "w"))

    ref = json.load(open(NATIVE_REF))
    same = sum(words(a) == words(b) for a, b in zip(texts, ref))
    print(f"MATCH vs native reference: {same}/{len(meta)}  (outputs saved to {out_path})")
    for i, (a, b) in enumerate(zip(texts, ref)):
        if words(a) != words(b):
            print(f"DIFF [{i:03d}]\n  this: {a}\n  ref : {b}")
    print(f"WER this run: {wer(meta, texts):.2%}   native reference: {wer(meta, ref):.2%}")


if __name__ == "__main__":
    main()
