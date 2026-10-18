import soundfile as sf
import torch
from safetensors.torch import load_file

from configuration_omniasr import OmniASRConfig
from modeling_omniasr import OmniAsrAudioTower, conv_output_length
from processing_omniasr import OmniASRFeatureExtractor


def set_attn_impl(cfg, impl):
    try:
        cfg._attn_implementation = impl
    except AttributeError:
        cfg._attn_implementation_internal = impl


cfg = OmniASRConfig.from_pretrained(".")
tower = OmniAsrAudioTower(cfg)

prefix = "model.audio_tower."
sd = load_file("model.safetensors")
sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
missing, unexpected = tower.load_state_dict(sub, strict=False)
print("loaded", len(sub), "tensors | missing:", missing, "| unexpected:", unexpected)
del sd

tower = tower.to("cuda", torch.float32).eval()
audio, sr = sf.read("/root/autodl-tmp/bench_audio/000.wav", dtype="float32")
x = OmniASRFeatureExtractor()(audio, sampling_rate=sr)["input_features"][0][None].to("cuda")

with torch.no_grad():
    set_attn_impl(cfg, "eager")
    out_eager = tower(x)
    set_attn_impl(cfg, "sdpa")
    out_sdpa = tower(x)

print("samples:", len(audio), "| output:", tuple(out_eager.shape),
      "| expected frames:", conv_output_length(cfg, len(audio)))
print("finite:", torch.isfinite(out_eager).all().item(),
      "| std %.4f" % out_eager.std().item())
print("eager vs sdpa: max abs diff %.2e" % (out_eager - out_sdpa).abs().max().item())
