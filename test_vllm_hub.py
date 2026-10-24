import soundfile as sf
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="ziywang50/omniASR-LLM-300M-hf",
        model_impl="transformers",
        trust_remote_code=True,
        dtype="bfloat16",
        attention_backend="TRITON_ATTN",   # decoder head_dim=512
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.75,
        limit_mm_per_prompt={"audio": 1},
    )
    audio, sr = sf.read("/root/autodl-tmp/bench_audio/000.wav", dtype="float32")
    out = llm.generate(
        {"prompt": "<pad><s>", "multi_modal_data": {"audio": (audio, sr)}},
        SamplingParams(temperature=0.0, max_tokens=256),
    )
    print("OUTPUT:", out[0].outputs[0].text)


if __name__ == "__main__":
    main()
