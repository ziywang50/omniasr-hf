"""Convert ziywang50/omniASR-LLM-300M (fairseq2-style names) to the
Transformers layout used by modeling_omniasr.py.

Usage: HF_HUB_OFFLINE=1 python convert_weights.py [out_path]
"""
import json
import re
import sys
from collections import Counter

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

SRC_REPO = "ziywang50/omniASR-LLM-300M"
# Tensor count of the 300M source checkpoint. Other variants (1B/3B/7B) have
# more encoder layers: update this tripwire when converting them.
EXPECTED_SRC = 535
POS_CONV = "encoder_frontend.pos_encoder.conv."

DECODER_RENAMES = {
    "self_attn_layer_norm.weight": "input_layernorm.weight",
    "ffn_layer_norm.weight": "post_attention_layernorm.weight",
    "self_attn.output_proj.weight": "self_attn.o_proj.weight",
    "ffn.gate_proj.weight": "mlp.gate_proj.weight",
    "ffn.inner_proj.weight": "mlp.up_proj.weight",
    "ffn.output_proj.weight": "mlp.down_proj.weight",
}


def permute_q_k_for_neox(w, num_heads):
    """GPT-J (interleaved pairs) -> NeoX (half split) RoPE layout, per head.
    Works for weights [out, in] and biases [out]."""
    out_dim = w.shape[0]
    head_dim = out_dim // num_heads
    rest = w.shape[1:]
    return w.view(num_heads, head_dim // 2, 2, *rest).transpose(1, 2).reshape(out_dim, *rest)


def unpermute_q_k_from_neox(w, num_heads):
    """Inverse of permute_q_k_for_neox (self-check only)."""
    out_dim = w.shape[0]
    head_dim = out_dim // num_heads
    rest = w.shape[1:]
    return w.view(num_heads, 2, head_dim // 2, *rest).transpose(1, 2).reshape(out_dim, *rest)


def fold_weight_norm(g, v):
    """weight_norm(dim=2): w = v * g / ||v||, norm over dims (0, 1) per kernel position."""
    return v * (g / v.norm(dim=(0, 1), keepdim=True))


def convert(name, tensor, rope_heads):
    """Return (new_name, new_tensor, num_heads_if_permuted_else_None)."""
    if name.startswith("encoder_frontend.") or name.startswith("encoder."):
        return "model.audio_tower." + name, tensor, None
    if name.startswith("encoder_proj."):
        return "model.multi_modal_projector." + name[len("encoder_proj."):], tensor, None
    if name.startswith("lang_embeddings."):
        return "model." + name, tensor, None
    if name == "text_frontend.weight":
        return "model.language_model.embed_tokens.weight", tensor, None
    if name == "llama_decoder.layer_norm.weight":
        return "model.language_model.norm.weight", tensor, None
    if name == "final_proj.weight":
        return "lm_head.weight", tensor, None
    m = re.fullmatch(r"llama_decoder\.layers\.(\d+)\.(.+)", name)
    if m:
        idx, rest = m.groups()
        heads = None
        parts = rest.split(".")
        if len(parts) > 1 and parts[0] == "self_attn" and parts[1] in rope_heads:   # q_proj / k_proj, weight or bias
            heads = rope_heads[parts[1]]
            tensor = permute_q_k_for_neox(tensor, heads)
        rest = DECODER_RENAMES.get(rest, rest)
        return f"model.language_model.layers.{idx}.{rest}", tensor, heads
    raise KeyError(f"Unmapped checkpoint key: {name}")


def group(key):
    p = key.split(".")
    if p[0] == "model" and p[1] in ("audio_tower", "language_model"):
        return ".".join(p[:3])
    return ".".join(p[:2])


def main(out_path):
    src_dir = snapshot_download(SRC_REPO, allow_patterns=["*.safetensors", "config.json"])
    src_cfg = json.load(open(f"{src_dir}/config.json"))
    text_cfg = src_cfg["text_config"]
    rope_heads = {
        "q_proj": text_cfg["num_attention_heads"],
        "k_proj": text_cfg["num_key_value_heads"],
    }
    print("rope heads from config:", rope_heads)

    out, n_permuted = {}, 0
    with safe_open(f"{src_dir}/model.safetensors", "pt") as f:
        names = list(f.keys())
        assert len(names) == EXPECTED_SRC, f"source has {len(names)} tensors, expected {EXPECTED_SRC}"

        # 1. weight_norm on the positional conv -> one plain conv weight
        g = f.get_tensor(POS_CONV + "weight_g")
        v = f.get_tensor(POS_CONV + "weight_v")
        w = fold_weight_norm(g, v)
        assert w.shape == v.shape
        assert torch.allclose(w.norm(dim=(0, 1)), g.abs().flatten(), rtol=1e-4), "weight_norm fold mismatch"
        out["model.audio_tower." + POS_CONV + "weight"] = w.contiguous()

        # 2. text_frontend -> embed_tokens (regular tokens) + lid_embedding (LID marker row),
        #    so embed_tokens and lm_head have the same vocab size
        emb = f.get_tensor("text_frontend.weight")
        lid = src_cfg["lid_marker_token_id"]
        assert lid == emb.shape[0] - 1, "LID marker must be the last embedding row"
        out["model.language_model.embed_tokens.weight"] = emb[:lid].clone()
        out["model.lid_embedding.weight"] = emb[lid:].clone()

        # 3. everything else
        for name in names:
            if name in (POS_CONV + "weight_g", POS_CONV + "weight_v", "text_frontend.weight"):
                continue
            orig = f.get_tensor(name)
            new_name, new, heads = convert(name, orig, rope_heads)
            if new_name in out:
                raise ValueError(f"Two source keys map to {new_name}")
            if heads is not None:   # check every permuted q/k tensor, every layer
                assert not torch.equal(new, orig), f"{name}: permutation was a no-op"
                assert torch.equal(unpermute_q_k_from_neox(new, heads), orig), f"{name}: not a pure row permutation"
                n_permuted += 1
            out[new_name] = new.contiguous()

    n_layers = text_cfg["num_hidden_layers"]
    assert len(out) == EXPECTED_SRC, f"{len(out)} converted, expected {EXPECTED_SRC}"
    assert n_permuted >= 2 * n_layers, f"permuted {n_permuted} tensors, expected >= {2 * n_layers}"

    save_file(out, out_path, metadata={"format": "pt"})
    print(f"converted {len(out)} tensors -> {out_path}  (q/k permuted: {n_permuted})")
    print("dtypes:", dict(Counter(str(t.dtype) for t in out.values())))
    for gname, n in sorted(Counter(group(k) for k in out).items()):
        print(f"  {gname}: {n}")
    print("self-checks passed")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model.safetensors")
