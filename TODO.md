# TODO (convert_weights.py, before converting 1B/3B/7B)
- Support sharded source checkpoints (model-0000X-of-0000N.safetensors + index.json).
- Use text_cfg.get("num_key_value_heads", text_cfg["num_attention_heads"]).
- Update EXPECTED_SRC for the variant (encoder layer count differs).
- processing_omniasr.py: replace deprecated feature_extractor_class = "AutoFeatureExtractor".
