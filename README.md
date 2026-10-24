---
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
datasets:
- facebook/omnilingual-asr-corpus
tags:
- speech-recognition
- omniasr
- custom_code
base_model: facebook/omniASR-LLM-300M
---

# omniASR-LLM-300M — Transformers port (remote code)

A Transformers-compatible port of Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
LLM 300M model ([facebook/omniASR-LLM-300M](https://huggingface.co/facebook/omniASR-LLM-300M)),
which is released for fairseq2 only. This repository adds Transformers modeling, config and
processor code, so the model loads with `AutoModelForCausalLM` / `AutoProcessor` and runs in
vLLM through its Transformers backend, with no changes to vLLM.

## Usage (Transformers)

```python
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

repo = "ziywang50/omniASR-LLM-300M-hf"
model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

audio, sr = sf.read("audio.wav", dtype="float32")          # 16 kHz mono, up to 40 s
inputs = processor(audio=audio, sampling_rate=sr, language="eng_Latn").to("cuda")
out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=1, eos_token_id=2)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

`language` accepts OmniASR codes (e.g. `eng_Latn`, `deu_Latn`) or ISO 639-1 codes for
common languages (e.g. `en`, `de`). The full list is in `lang_lookup.json`.

`AutoModel` loads the base model without the LM head (the convention the vLLM Transformers
backend relies on); use `AutoModelForCausalLM` for generation.

## Usage (vLLM)

```python
import soundfile as sf
from vllm import LLM, SamplingParams

llm = LLM(
    model="ziywang50/omniASR-LLM-300M-hf",
    model_impl="transformers",
    trust_remote_code=True,
    attention_backend="TRITON_ATTN",   # the decoder's head_dim is 512
    limit_mm_per_prompt={"audio": 1},
)
audio, sr = sf.read("audio.wav", dtype="float32")          # 16 kHz mono, up to 40 s
out = llm.generate(
    {"prompt": "<pad><s>", "multi_modal_data": {"audio": (audio, sr)},
     "mm_processor_kwargs": {"language": "eng_Latn"}},
    SamplingParams(temperature=0.0, max_tokens=256),
)
print(out[0].outputs[0].text)
```

The prompt is always `"<pad><s>"`: the processor expands `<pad>` into the audio
placeholders, and `<s>` is BOS.

## Verification

Evaluated on the 73 clips of `hf-internal-testing/librispeech_asr_dummy` (clean/validation),
bf16, greedy decoding. The two vLLM rows were measured on one RTX 4090 in the same session:
the 73 clips repeated 5x (365 requests), prefix caching and the multimodal processor cache
disabled, median of 5 runs after 2 warm-up runs
([bench/bench_vllm.py](https://github.com/ziywang50/omniasr-hf/blob/master/bench/bench_vllm.py)).

| | RTFx | Transcripts identical to native vLLM | WER |
|---|---|---|---|
| vLLM, Transformers backend + this repo | 220.0 | 73 / 73 | 7.04% |
| vLLM, native implementation ([vllm#36864](https://github.com/vllm-project/vllm/pull/36864), not merged) | 227.0 | (reference) | 7.04% |
| Transformers `generate`, one clip at a time | 6.3¹ | 73 / 73 | 7.04% |
| Meta `omnilingual-asr` reference (fairseq2, batch size 16) | 7.0¹ | 69 / 73² | 7.04% |

RTFx = seconds of audio transcribed per second of wall-clock time.
¹ Measured in separate sessions on an RTX 4090, so not a like-for-like throughput comparison.
² The 4 differences are single words; WER is identical.

## What the conversion changes

The weights are a mathematically equivalent re-layout of the original checkpoint
(see `convert_weights.py`):

- Parameter names are mapped from fairseq2 to a Transformers layout; the decoder is a
  standard `LlamaModel`.
- Decoder `q_proj` / `k_proj` rows are permuted from the GPT-J (interleaved) RoPE layout to
  the NeoX (half-split) layout used by Transformers' Llama, so the standard RoPE reproduces
  the original model.
- The `weight_norm` on the encoder's positional convolution is folded into a plain
  `Conv1d` weight.
- The LID-marker row of the original text embedding is stored as a separate
  `lid_embedding` parameter, so `embed_tokens` and `lm_head` have the same vocabulary size
  (9812), as the vLLM Transformers backend requires.

## Limitations

- One audio clip per processor call: `<pad>` is both the audio placeholder and the
  tokenizer's padding token, so padding a batch of prompts would create extra placeholders.
  vLLM batches requests itself and is not affected.
- Clips up to 40 s, as for the original LLM models. Long-form audio needs the Unlimited
  variants, which are not ported.
- Only the 300M checkpoint has been converted, and only English was evaluated.
- vLLM: tested with the offline `LLM` API only (not `vllm serve`), at vLLM commit
  `32fbfa15e8`, with `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

Tested with `transformers==5.15.1`, `torch==2.13.0`.

## Credits

The model was developed by Meta FAIR (Omnilingual ASR). This repository only repackages the
weights and adds Transformers code. Parts of the modeling and conversion code were written
with AI assistance (Claude); all results above come from actual runs.

License: Apache-2.0, same as the original release.
