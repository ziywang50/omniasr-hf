import json, re, sys
import editdistance

def norm(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()

meta = json.load(open("/root/autodl-tmp/bench_audio/manifest.json"))
ref_out = json.load(open(sys.argv[1]))     # e.g. /root/autodl-tmp/ref_out_bs16.json
vllm_out = json.load(open(sys.argv[2]))    # e.g. /root/autodl-tmp/vllm_out.json

same = sum(norm(a) == norm(b) for a, b in zip(ref_out, vllm_out))
print(f"vLLM vs reference identical: {same}/{len(meta)}")
for i, (a, b) in enumerate(zip(ref_out, vllm_out)):
    if norm(a) != norm(b):
        print(f"  [{i:03d}] ref : {a}\n        vllm: {b}")

def wer(hyps):
    err = sum(editdistance.eval(norm(m["ref"]), norm(h)) for m, h in zip(meta, hyps))
    return err / sum(len(norm(m["ref"])) for m in meta)

print(f"WER reference: {wer(ref_out):.2%}   WER vLLM: {wer(vllm_out):.2%}")
