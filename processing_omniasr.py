# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm-project/vllm#36864 (same author).
import json

import numpy as np
import torch
from transformers import BatchFeature
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin
from transformers.utils import cached_file

NORMALIZATION_EPS = 1e-7
AUDIO_TOKEN = "<pad>"   # id 1: audio placeholder (also the tokenizer's padding token)
BOS_TOKEN_ID = 0        # "<s>"; the model checks bos_token_id == 0
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
    model_input_names = ["input_features", "input_features_mask"]

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
        """Frames after the CNN, ceil-divided by encoder_stacking.

        Simplified Conv1d output-length formula: assumes padding=0 and dilation=1
        for every conv layer (true for OmniASR). Must match
        modeling_omniasr.num_audio_tokens.
        """
        n = int(num_samples)
        for _out_ch, kernel_size, stride in self.conv_layer_descs:
            n = (n - kernel_size) // stride + 1
        return -(-n // self.encoder_stacking)

    def __call__(self, audio, sampling_rate=None, **kwargs) -> BatchFeature:
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"Expected sampling_rate={self.sampling_rate}, got {sampling_rate}. "
                "OmniASR requires 16kHz audio."
            )
        if not isinstance(audio, (list, tuple)):
            audio = [audio]
        clips = []
        for a in audio:
            t = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
            clips.append((t - t.mean()) / torch.sqrt(t.var() + NORMALIZATION_EPS))
        # Pad to the longest clip; the mask records each clip's true length.
        max_len = max(c.shape[0] for c in clips)
        features = torch.zeros(len(clips), max_len)
        mask = torch.zeros(len(clips), max_len, dtype=torch.long)
        for i, c in enumerate(clips):
            features[i, : c.shape[0]] = c
            mask[i, : c.shape[0]] = 1
        return BatchFeature(data={"input_features": features, "input_features_mask": mask})


class OmniASRAudioKwargs(AudioKwargs, total=False):
    language: str | list[str]


class OmniASRProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: OmniASRAudioKwargs
    _defaults = {
        "text_kwargs": {"add_special_tokens": False},   # BOS is written in the prompt
        "audio_kwargs": {"sampling_rate": 16000, "language": "eng_Latn"},
        "common_kwargs": {"return_tensors": "pt"},
    }


class OmniASRProcessor(ProcessorMixin):
    """Prompt: "<pad><s>". The "<pad>" expands to N + 2 placeholders (N audio
    frames, then the LID marker and language slots), which the model fills;
    "<s>" is BOS.

    One clip per call: "<pad>" is also the tokenizer's padding token, so padding
    a batch of prompts would create extra placeholders.
    """

    valid_processor_kwargs = OmniASRProcessorKwargs
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"
    n_special_tokens = 1    # TODO: read from config

    def __init__(self, feature_extractor, tokenizer, audio_token=AUDIO_TOKEN):
        self.audio_token = audio_token
        self.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)
        self.bos_token_str = tokenizer.convert_ids_to_tokens(BOS_TOKEN_ID)
        self.lang_table = None
        super().__init__(feature_extractor, tokenizer)

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

    def __call__(self, text=None, audio=None, **kwargs):
        if audio is not None:
            n = len(audio) if isinstance(audio, (list, tuple)) else 1
            if n != 1:
                raise NotImplementedError(
                    "OmniASRProcessor takes one clip per call: '<pad>' is both the audio "
                    "placeholder and the tokenizer's padding token."
                )
            if text is None:
                text = self.audio_token + self.bos_token_str
        return super().__call__(text=text, audio=audio, **kwargs)

    def _process_audio(self, audio, **kwargs):
        language = kwargs.pop("language", "eng_Latn")
        clips = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        audio_inputs = self.feature_extractor(clips, **kwargs)
        lengths = audio_inputs["input_features_mask"].sum(-1).tolist()
        audio_inputs["num_audio_tokens"] = [
            self.feature_extractor.num_audio_tokens(n) + 2 for n in lengths   # + LID + language
        ]
        langs = language if isinstance(language, (list, tuple)) else [language] * len(clips)
        audio_inputs["language_ids"] = torch.tensor([self.language_id(l) for l in langs])
        replacements = [self.replace_audio_token(audio_inputs, audio_idx=i) for i in range(len(clips))]
        return audio_inputs, replacements

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int, **kwargs) -> str:
        return self.audio_token * int(audio_inputs["num_audio_tokens"][audio_idx])

    @property
    def model_input_names(self) -> list[str]:
        names = list(super().model_input_names)
        return names + [n for n in ("input_features_mask", "language_ids") if n not in names]

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_audio_tokens"]

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)
