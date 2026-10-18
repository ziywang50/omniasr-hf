import json
import time
import time
import re

import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor


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


d = "/root/autodl-tmp/omniasr-hf"
model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
proc = AutoProcessor.from_pretrained(d, trust_remote_code=True)
meta = json.load(open("/root/autodl-tmp/bench_audio/manifest.json"))
vllm_out = json.load(open("/root/autodl-tmp/vllm_out.json"))

hf_out, same = [], 0
torch.cuda.synchronize(); t0 = time.perf_counter()
torch.cuda.synchronize(); t0 = time.perf_counter()
for i, m in enumerate(meta):
    audio, sr = sf.read(m["path"], dtype="float32")
    inputs = proc(audio=audio, sampling_rate=sr, language="eng_Latn").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False,
                             pad_token_id=1, eos_token_id=2)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    hf_out.append(text)
    if words(text) == words(vllm_out[i]):
        same += 1
    else:
        print(f"[{i:03d}] hf  : {text}\n      vllm: {vllm_out[i]}")

torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
total_dur = sum(m['dur'] for m in meta)
print(f'Transformers (bf16, one clip at a time): {elapsed:.1f}s for {total_dur:.0f}s audio, RTFx {total_dur/elapsed:.1f}')

torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
total_dur = sum(m['dur'] for m in meta)
print(f'Transformers (bf16, one clip at a time): {elapsed:.1f}s for {total_dur:.0f}s audio, RTFx {total_dur/elapsed:.1f}')

def wer(hyps):
    err = sum(edit_distance(words(m["ref"]), words(h)) for m, h in zip(meta, hyps))
    return err / sum(len(words(m["ref"])) for m in meta)

print(f"\nTransformers vs native vLLM identical: {same}/{len(meta)}")
print(f"WER  Transformers: {wer(hf_out):.2%}   native vLLM: {wer(vllm_out):.2%}")
json.dump(hf_out, open("/root/autodl-tmp/hf_out.json", "w"))
