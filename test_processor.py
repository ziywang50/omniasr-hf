import soundfile as sf
from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained(".", trust_remote_code=True)
audio, sr = sf.read("/root/autodl-tmp/bench_audio/000.wav", dtype="float32")

a = proc(audio=audio, sampling_rate=sr, language="eng_Latn")
b = proc(text="<pad><s>", audio=audio, sampling_rate=sr, language="eng_Latn",
         return_text_replacement_offsets=True)

n = proc.feature_extractor.num_audio_tokens(len(audio)) + 2
ids = a["input_ids"][0].tolist()
print("keys:", sorted(a.keys()))
print("ids == [1]*(N+2) + [0]:", ids == [1] * n + [0], "| len", len(ids), "| N+2 =", n)
print("explicit prompt gives the same ids:", b["input_ids"].tolist() == a["input_ids"].tolist())
print("offsets:", [(o["type"], o["span"], len(o["replacement"])) for o in b["text_replacement_offsets"][0]])
print("language_ids:", a["language_ids"].tolist(),
      "| features", tuple(a["input_features"].shape), "| mask sum", int(a["input_features_mask"].sum()))
