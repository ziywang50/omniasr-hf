# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm#36864 (same author).
import json

import numpy as np
import torch
from transformers import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import cached_file

NORMALIZATION_EPS = 1e-7
DEFAULT_CONV_LAYER_DESCS = [[512, 10, 5], [512, 3, 2], [512, 3, 2], [512, 3, 2],
                            [512, 3, 2], [512, 2, 2], [512, 2, 2]]
ISO_TO_OMNIASR_CODE = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "zh": "cmn_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "ru": "rus_Cyrl",
    "ar": "arb_Arab", "pt": "por_Latn", "it": "ita_Latn", "hi": "hin_Deva",
    "tr": "tur_Latn", "pl": "pol_Latn", "nl": "nld_Latn", "vi": "vie_Latn",
    "th": "tha_Thai", "uk": "ukr_Cyrl", "id": "ind_Latn", "fi": "fin_Latn",
    "sv": "swe_Latn", "cs": "ces_Latn", "el": "ell_Grek", "he": "heb_Hebr",
    "ro": "ron_Latn",
}


class OmniASRFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features"]

    def __init__(self, sampling_rate=16000, padding_value=0.0, feature_size=1,
                 conv_layer_descs=None, encoder_stacking=1, **kwargs):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.conv_layer_descs = conv_layer_descs or DEFAULT_CONV_LAYER_DESCS
        self.encoder_stacking = encoder_stacking

    def num_audio_tokens(self, num_samples: int) -> int:
        """Same formula as modeling_omniasr.num_audio_tokens (frames after the
        CNN, then ceil-divided by encoder_stacking)."""
        n = num_samples
        for _out_ch, kernel_size, stride in self.conv_layer_descs:
            n = (n - kernel_size) // stride + 1
        return -(-n // self.encoder_stacking)

    def __call__(self, audio, sampling_rate=None, **kwargs) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}. "
                "OmniASR requires 16kHz audio."
            )
        if not isinstance(audio, list):
            audio = [audio]
        features = []
        for a in audio:
            t = torch.as_tensor(np.asarray(a), dtype=torch.float32)
            t = (t - t.mean()) / torch.sqrt(t.var() + NORMALIZATION_EPS)
            features.append(t)
        return BatchFeature(data={"input_features": features})


class OmniASRProcessor(ProcessorMixin):
    """Builds [<pad> x (N + 2), <s>]: N audio frames, then the LID marker and
    language embedding slots, then BOS. The model fills the <pad> slots."""

    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    audio_token_id = 1      # <pad>
    bos_token_id = 0
    n_special_tokens = 1

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.lang_table = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        with open(cached_file(pretrained_model_name_or_path, "lang_lookup.json")) as f:
            processor.lang_table = json.load(f)
        return processor

    def language_id(self, language: str) -> int:
        code = ISO_TO_OMNIASR_CODE.get(language, language)
        if self.lang_table is None or code not in self.lang_table:
            raise ValueError(f"Unsupported language {language!r}: use ISO-639-1 ('en') "
                             "or an OmniASR code ('eng_Latn').")
        return self.lang_table[code] + self.n_special_tokens

    def __call__(self, audio=None, language="eng_Latn", sampling_rate=None, **kwargs):
        if audio is None:
            raise ValueError("audio is required")
        clips = audio if isinstance(audio, list) else [audio]
        if len(clips) != 1:
            raise NotImplementedError("one clip per call for now (no encoder padding mask)")
        feats = self.feature_extractor(clips, sampling_rate=sampling_rate)["input_features"]
        n = self.feature_extractor.num_audio_tokens(feats[0].shape[-1]) + 2  # + LID + language
        input_ids = torch.tensor([[self.audio_token_id] * n + [self.bos_token_id]])
        return BatchFeature(data={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "input_features": torch.stack(feats),
            "language_ids": torch.tensor([self.language_id(language)]),
        })

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
