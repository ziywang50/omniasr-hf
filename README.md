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
processor code so the model loads with `AutoModel` / `AutoProcessor`.

## Usage

```python
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor

repo = "ziywang50/omniASR-LLM-300M-hf"
model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

audio, sr = sf.read("audio.wav", dtype="float32")          # 16 kHz mono, up to 40 s
inputs = processor(audio=audio, sampling_rate=sr, language="eng_Latn").to("cuda")
out = model.generate(**inputs, max_new_tokens=256, do_sample=False, pad_token_id=1, eos_token_id=2)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

`language` accepts OmniASR codes (e.g. `eng_Latn`, `deu_Latn`) or ISO 639-1 codes for
common languages (e.g. `en`, `de`). The full list is in `lang_lookup.json`.

## Verification

Evaluated on the 73 clips of `hf-internal-testing/librispeech_asr_dummy` (clean/validation),
bf16, greedy decoding:

| | Result |
|---|---|
| Transcripts identical to the native vLLM implementation ([vllm#36864](https://github.com/vllm-project/vllm/pull/36864)) | 73 / 73 |
| WER, this port | 7.04% |
| WER, native vLLM implementation | 7.04% |

The native vLLM implementation was itself validated against Meta's `omnilingual-asr`
reference pipeline on the same clips (WER 7.04% for both).

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

## Limitations

- One audio clip per call (the encoder has no padding mask yet).
- Clips up to 40 s, as for the original LLM models. Long-form audio needs the Unlimited
  variants, which are not ported.
- Only English was evaluated here.
- vLLM support through the Transformers backend is work in progress.

Tested with `transformers==5.15.1`, `torch==2.13.0`.

## Credits

The model was developed by Meta FAIR (Omnilingual ASR). This repository only repackages the
weights and adds Transformers code. Parts of the modeling and conversion code were written
with AI assistance (Claude); all results above come from actual runs.

License: Apache-2.0, same as the original release.
