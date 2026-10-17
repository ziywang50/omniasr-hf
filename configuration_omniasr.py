# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm#36864 (same author).
from transformers import LlamaConfig, PreTrainedConfig


class OmniASRConfig(PreTrainedConfig):
    """Configuration for OmniASR LLM models (wav2vec2 encoder + LLaMA decoder).

    Defaults correspond to the 300M encoder variant. The LLaMA decoder is
    shared across all variants (300M, 1B, 3B, 7B).
    """

    model_type = "omniasr_llm"
    sub_configs = {"text_config": LlamaConfig}

    def __init__(
        self,
        sampling_rate: int = 16000,
        subsampling_factor: int = 320,
        feature_dim: int = 512,
        feature_extractor_layer_descs: list | None = None,
        feature_extractor_bias: bool = True,
        feature_extractor_layer_norm_convs: bool = True,
        encoder_embed_dim: int = 1024,
        encoder_num_heads: int = 16,
        encoder_ffn_dim: int = 4096,
        encoder_num_layers: int = 24,
        pos_encoder_kernel_size: int = 128,
        pos_encoder_groups: int = 16,
        projection_dim: int | None = None,
        text_config: dict | LlamaConfig | None = None,
        lang_embeddings_p: float = 0.5,
        n_special_tokens: int = 1,
        target_vocab_size: int = 9812,
        num_languages: int = 1694,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        pad_token_id: int = 1,
        lid_marker_token_id: int = 9812,
        encoder_stacking: int = 1,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.subsampling_factor = subsampling_factor
        self.feature_dim = feature_dim
        self.feature_extractor_layer_descs = feature_extractor_layer_descs or [
            [512, 10, 5], [512, 3, 2], [512, 3, 2], [512, 3, 2],
            [512, 3, 2], [512, 2, 2], [512, 2, 2],
        ]
        self.feature_extractor_bias = feature_extractor_bias
        self.feature_extractor_layer_norm_convs = feature_extractor_layer_norm_convs
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_num_heads = encoder_num_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_num_layers = encoder_num_layers
        self.pos_encoder_kernel_size = pos_encoder_kernel_size
        self.pos_encoder_groups = pos_encoder_groups

        if text_config is None:
            text_config = {
                "hidden_size": 4096,
                "num_attention_heads": 8,
                "num_key_value_heads": 8,
                "num_hidden_layers": 12,
                "intermediate_size": 2816,
                "max_position_embeddings": 8192,
                "rope_theta": 10000.0,
                "vocab_size": 9813,
                "rms_norm_eps": 1e-05,
            }
        if isinstance(text_config, dict):
            text_config = LlamaConfig(**text_config)
        self.text_config = text_config

        self.projection_dim = projection_dim or self.text_config.hidden_size
        self.lang_embeddings_p = lang_embeddings_p
        self.n_special_tokens = n_special_tokens
        self.target_vocab_size = target_vocab_size
        self.num_languages = num_languages
        self.lid_marker_token_id = lid_marker_token_id
        self.encoder_stacking = encoder_stacking

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
