# TODO (convert_weights.py, before converting 1B/3B/7B)
- Support sharded source checkpoints (model-0000X-of-0000N.safetensors + index.json).
- Use text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"]).
- Update EXPECTED_SRC for the variant (encoder layer count differs).
- processing_omniasr.py: replace deprecated feature_extractor_class = "AutoFeatureExtractor".
- Processor: read n_special_tokens from config instead of hard-coding 1.
- Processor: conv_layer_descs / encoder_stacking come from preprocessor_config; keep in sync with config.json.
- Processor: warn on clips longer than 40 s.
- Native vLLM PR: embed_multimodal uses language_id.flatten()[0] for the whole batch; check mixed-language batches.
