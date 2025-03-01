from vllm import LLM

from gen_params import LLMS, SAMPLING_PARAMS, SEED

PROMPT = [
    {
        "role": "system",
        "content": "You are a helpful assistant for answering questions.",
    },
    {"role": "user", "content": "What is the capital of Poland?"},
]


if __name__ == "__main__":
    for llm, quant, path in LLMS:
        print(f"Testing {llm}...")
        llm = LLM(model=path, quantization=quant, trust_remote_code=True, seed=SEED)
        response = llm.chat(PROMPT, sampling_params=SAMPLING_PARAMS[0], use_tqdm=False)
        print(response)
        print()
