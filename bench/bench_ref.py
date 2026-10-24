import json, statistics, sys, time
import soundfile as sf
import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

REPEAT, WARMUP, RUNS = 5, 2, 5
BS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
meta = json.load(open("/root/autodl-tmp/bench_audio/manifest.json"))
audios = [{"waveform": sf.read(m["path"], dtype="float32")[0], "sample_rate": 16000} for m in meta]
inp = audios * REPEAT                      # 重复引用同一批波形，不复制数据
total_dur = sum(len(a["waveform"]) / 16000 for a in inp)

p = ASRInferencePipeline(model_card="omniASR_LLM_300M")
for _ in range(WARMUP):                    # 用完整 batch warmup
    p.transcribe(inp[:BS], lang=["eng_Latn"] * BS, batch_size=BS)

torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
times = []
for run in range(RUNS):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = p.transcribe(inp, lang=["eng_Latn"] * len(inp), batch_size=BS)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
assert len(out) == len(inp), f"expected {len(inp)} outputs, got {len(out)}"

med = statistics.median(times)
print(f"REF bs={BS} median={med:.2f}s RTFx={total_dur / med:.1f} "
      f"runs={[round(t, 2) for t in times]} peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
json.dump(out[:len(meta)], open(f"/root/autodl-tmp/ref_out_bs{BS}.json", "w"))
