import json

import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

d = "/root/autodl-tmp/omniasr-hf"
model, info = AutoModel.from_pretrained(d, trust_remote_code=True, dtype=torch.bfloat16,
                                        output_loading_info=True)
print(type(model).__name__, "| loading issues:", {k: v for k, v in info.items() if v} or "none")
model = model.to("cuda").eval()
proc = AutoProcessor.from_pretrained(d, trust_remote_code=True)

ref = json.load(open("/root/autodl-tmp/vllm_out.json"))
for i in range(3):
    audio, sr = sf.read(f"/root/autodl-tmp/bench_audio/{i:03d}.wav", dtype="float32")
    inputs = proc(audio=audio, sampling_rate=sr, language="eng_Latn").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False,
                             pad_token_id=1, eos_token_id=2)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    same = text.strip() == ref[i].strip()
    print(f"[{i}] {'MATCH' if same else 'DIFF '} | {text}")
    if not same:
        print(f"      vllm  | {ref[i]}")
