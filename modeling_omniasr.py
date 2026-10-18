# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm#36864 (same author).
import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

try:
    from .configuration_omniasr import OmniASRConfig
except ImportError:  # running as a plain script inside the repo directory
    from configuration_omniasr import OmniASRConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def conv_output_length(config: OmniASRConfig, num_samples: int) -> int:
    """Frames after the CNN feature extractor (Conv1d, padding=0, dilation=1)."""
    n = num_samples
    for _out_ch, kernel_size, stride in config.feature_extractor_layer_descs:
        n = (n - kernel_size) // stride + 1
    return n


def num_audio_tokens(config: OmniASRConfig, num_samples: int) -> int:
    """Decoder positions taken by one audio clip, after encoder_stacking (excl. lid/lang)."""
    frames = conv_output_length(config, num_samples)
    return -(-frames // config.encoder_stacking)  # ceil division


def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value).transpose(1, 2).contiguous()
    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Wav2Vec2 audio encoder (parameter names match the original checkpoint)
# ---------------------------------------------------------------------------
class OmniAsrConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias, layer_norm):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, bias=bias)
        self.layer_norm = nn.LayerNorm(out_ch) if layer_norm else None

    def forward(self, x):  # [B, C_in, T] -> [B, C_out, T']
        x = self.conv(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return F.gelu(x)


class OmniAsrFeatureExtractor(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch, kernel_size, stride in config.feature_extractor_layer_descs:
            layers.append(OmniAsrConvLayer(
                in_ch, out_ch, kernel_size, stride,
                config.feature_extractor_bias, config.feature_extractor_layer_norm_convs,
            ))
            in_ch = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class OmniAsrPositionalConv(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        k = config.pos_encoder_kernel_size
        self.conv = nn.Conv1d(
            config.encoder_embed_dim, config.encoder_embed_dim,
            kernel_size=k, padding=k // 2, groups=config.pos_encoder_groups, bias=True,
        )


class OmniAsrFrontend(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.feature_extractor = OmniAsrFeatureExtractor(config)
        self.post_extract_layer_norm = nn.LayerNorm(config.feature_dim)
        self.model_dim_proj = nn.Linear(config.feature_dim, config.encoder_embed_dim, bias=True)
        self.pos_encoder = OmniAsrPositionalConv(config)

    def forward(self, audio):  # [B, num_samples]
        x = audio.unsqueeze(1) if audio.dim() == 2 else audio
        x = self.feature_extractor(x)          # [B, feature_dim, T]
        x = x.transpose(1, 2)                  # [B, T, feature_dim]
        x = self.post_extract_layer_norm(x)
        x = self.model_dim_proj(x)             # [B, T, E]
        pos = self.pos_encoder.conv(x.transpose(1, 2))
        pos = pos[:, :, : x.shape[1]]          # even kernel -> one extra frame
        return x + F.gelu(pos).transpose(1, 2)


class OmniAsrEncoderAttention(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.config = config
        embed_dim = config.encoder_embed_dim
        self.num_heads = config.encoder_num_heads
        self.head_dim = embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.is_causal = False  # bidirectional; sdpa assumes causal if this is missing
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        impl = getattr(self.config, "_attn_implementation", None) or "eager"
        attn_fn = eager_attention_forward if impl == "eager" else ALL_ATTENTION_FUNCTIONS[impl]
        out, _ = attn_fn(self, q, k, v, attention_mask=None, dropout=0.0,
                         scaling=self.scaling, is_causal=False)
        return self.output_proj(out.reshape(bsz, seq_len, -1).contiguous())


class OmniAsrEncoderFFN(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.inner_proj = nn.Linear(config.encoder_embed_dim, config.encoder_ffn_dim, bias=True)
        self.output_proj = nn.Linear(config.encoder_ffn_dim, config.encoder_embed_dim, bias=True)

    def forward(self, x):
        return self.output_proj(F.gelu(self.inner_proj(x)))


class OmniAsrEncoderLayer(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        self.self_attn = OmniAsrEncoderAttention(config)
        self.ffn_layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        self.ffn = OmniAsrEncoderFFN(config)

    def forward(self, x):
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        return x + self.ffn(self.ffn_layer_norm(x))


class OmniAsrEncoder(nn.Module):
    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.layers = nn.ModuleList([OmniAsrEncoderLayer(config) for _ in range(config.encoder_num_layers)])
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)


class OmniAsrAudioTower(nn.Module):
    """encoder_frontend + encoder. Returns [B, T, encoder_embed_dim]."""

    def __init__(self, config: OmniASRConfig):
        super().__init__()
        self.encoder_frontend = OmniAsrFrontend(config)
        self.encoder = OmniAsrEncoder(config)

    def forward(self, audio):
        audio = audio.to(self.encoder_frontend.model_dim_proj.weight.dtype)
        return self.encoder(self.encoder_frontend(audio))


# ---------------------------------------------------------------------------
# Full model: audio tower + projector + LID/language embeddings + LLaMA decoder
# ---------------------------------------------------------------------------
from transformers import GenerationMixin, LlamaModel, PreTrainedModel  # noqa: E402
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: E402


class OmniAsrPreTrainedModel(PreTrainedModel):
    config_class = OmniASRConfig
    base_model_prefix = "model"
    _supports_sdpa = True
    _supports_flash_attn = False      # decoder head_dim is 512
    _supports_flash_attn_2 = False
    _supports_attention_backend = True
    _no_split_modules = ["OmniAsrEncoderLayer", "LlamaDecoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class OmniAsrModel(OmniAsrPreTrainedModel):
    def __init__(self, config: OmniASRConfig):
        super().__init__(config)
        if config.pad_token_id != 1 or config.bos_token_id != 0:
            raise ValueError("OmniASR expects pad_token_id=1 (audio placeholder) and bos_token_id=0")
        self.audio_tower = OmniAsrAudioTower(config)
        self.multi_modal_projector = nn.Linear(
            config.encoder_embed_dim * config.encoder_stacking, config.projection_dim, bias=True
        )
        self.lang_embeddings = nn.Embedding(config.num_languages, config.text_config.hidden_size)
        # LID marker embedding, kept outside embed_tokens so that embed_tokens and
        # lm_head share the same vocab size (required by the vLLM Transformers backend).
        self.lid_embedding = nn.Embedding(1, config.text_config.hidden_size)
        self.language_model = LlamaModel(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def get_audio_features(self, input_features, language_ids):
        """One clip at a time (the encoder has no padding mask).
        Returns a list of [N_i + 2, hidden]: audio frames, LID marker, language."""
        if input_features.dim() == 1:
            input_features = input_features[None]
        stacking = self.config.encoder_stacking
        outputs = []
        for i, wav in enumerate(input_features):
            x = self.audio_tower(wav[None])                       # [1, T, E]
            if stacking > 1:
                pad = (-x.size(1)) % stacking
                if pad:
                    x = F.pad(x, (0, 0, 0, pad))
                x = x.reshape(1, x.size(1) // stacking, x.size(2) * stacking)
            x = self.multi_modal_projector(x)[0]                  # [N, hidden]
            lid = self.lid_embedding.weight                       # [1, hidden]
            lang = self.lang_embeddings(language_ids[i].reshape(1).to(x.device))
            outputs.append(torch.cat([x, lid.to(x.dtype), lang.to(x.dtype)], dim=0))
        return outputs

    def forward(self, input_ids=None, input_features=None, language_ids=None,
                attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, cache_position=None, use_cache=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)
        if input_features is not None:
            audio = torch.cat(self.get_audio_features(input_features, language_ids), dim=0)
            mask = input_ids == self.config.pad_token_id          # audio placeholder slots
            if int(mask.sum()) != audio.shape[0]:
                raise ValueError(f"{int(mask.sum())} placeholder tokens but "
                                 f"{audio.shape[0]} audio embeddings")
            inputs_embeds = inputs_embeds.masked_scatter(
                mask.unsqueeze(-1).expand_as(inputs_embeds), audio.to(inputs_embeds.dtype)
            )
        return self.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            cache_position=cache_position, use_cache=use_cache, **kwargs,
        )


class OmniAsrForConditionalGeneration(OmniAsrPreTrainedModel, GenerationMixin):
    def __init__(self, config: OmniASRConfig):
        super().__init__(config)
        self.model = OmniAsrModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.target_vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_audio_features(self, input_features, language_ids):
        return self.model.get_audio_features(input_features, language_ids)

    def forward(self, input_ids=None, input_features=None, language_ids=None,
                attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, cache_position=None, use_cache=None,
                logits_to_keep=0, **kwargs):
        out = self.model(
            input_ids=input_ids, input_features=input_features, language_ids=language_ids,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds,
            cache_position=cache_position, use_cache=use_cache, **kwargs,
        )
        hidden = out.last_hidden_state
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, keep, :])
        return CausalLMOutputWithPast(logits=logits, past_key_values=out.past_key_values)

    def prepare_inputs_for_generation(self, *args, input_features=None, language_ids=None, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        cache = model_inputs.get("past_key_values")
        first_step = cache is None or not hasattr(cache, "get_seq_length") or cache.get_seq_length() == 0
        if first_step:   # audio only on the prefill step; later steps reuse the KV cache
            model_inputs["input_features"] = input_features
            model_inputs["language_ids"] = language_ids
        return model_inputs
