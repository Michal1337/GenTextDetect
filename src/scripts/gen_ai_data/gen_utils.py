import csv
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from gen_params import API_KEY, MAX_MODEL_LEN, MAX_RETRIES, MAX_WORKERS, SEED

random.seed(SEED)


def batchify(iterable: Iterable[str], batch_size: int):
    """Splits an iterable into smaller batches."""
    iterable = iter(iterable)
    while batch := list(islice(iterable, batch_size)):
        yield batch


def save_to_csv(
    path: str,
    prompts: List[str],
    responses: List[str],
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    """Saves prompts, responses and sampling parameters to a CSV file."""
    with open(path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        for prompt, response in zip(prompts, responses):
            writer.writerow([prompt, response, temperature, top_p, top_k])


def generate_responses(
    model: LLM, prompts: List[str], sampling_params: SamplingParams
) -> List[str]:
    """Generate a batch of outputs using vLLM with customizable sampling parameters."""
    outputs = model.chat(
        prompts,
        sampling_params=sampling_params,
        add_generation_prompt=False,
        continue_final_message=True,
        use_tqdm=False,
    )

    return [preproc_response(sample.outputs[0].text) for sample in outputs]


def preproc_response(s: str) -> str:
    """
    Removes single or double quotes from the start and end of the string if they exist and removes leading newlines and spaces.
    """
    s = s.lstrip("\n ")

    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def retry_request(
    client: OpenAI,
    prompt: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    retries: int,
) -> Union[str, None]:
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=prompt,
                temperature=temperature,
                top_p=top_p,
            )
            return preproc_response(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Retry {attempt+1} failed: {e}")
            time.sleep(random.uniform(0, 2))
    # final failure
    return None


def generate_responses_gpt(
    client: OpenAI,
    prompts: List[List[Dict[str, str]]],
    temperature: float,
    top_p: float,
) -> List[str]:
    # executor.map returns results in the same order as the inputs
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        responses = list(
            executor.map(
                lambda p: retry_request(client, p, temperature, top_p, MAX_RETRIES),
                prompts,
            )
        )

    # separate successes from failures
    results = [r for r in responses if r is not None]
    failed = [prompts[i] for i, r in enumerate(responses) if r is None]

    # final retry pass (also preserving order among the failures if you care)
    if failed:
        print(f"Retrying {len(failed)} failed prompts...")
        retry_responses = []
        for prompt in tqdm(failed, desc="Retrying failed"):
            retry_responses.append(
                retry_request(client, prompt, temperature, top_p, MAX_RETRIES)
            )

        # stitch retries back into the original order:
        for idx, resp in zip([prompts.index(p) for p in failed], retry_responses):
            if resp is not None:
                results.insert(idx, resp)
            else:
                failed.append(prompts[idx])

    return results


def check_for_too_long_prompts(
    df: pd.DataFrame, prompts: List[List[Dict[str, str]]], max_tokens_prompt: int
) -> Tuple[pd.DataFrame, List[List[Dict[str, str]]]]:
    """Check if any of the prompts are too long."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    lens = []
    batch_size = 128

    print("Tokenizing prompts...")
    for prompts_batch in tqdm(
        batchify(prompts, batch_size), total=len(prompts) // batch_size
    ):
        tokens = tokenizer.apply_chat_template(prompts_batch)
        lens.extend([len(token) for token in tokens])

    too_long = [idx for idx, length in enumerate(lens) if length > max_tokens_prompt]
    df.drop(too_long, inplace=True)
    prompts = [prompts[i] for i in range(len(prompts)) if i not in too_long]
    print("Removed too long prompts:", len(too_long))

    return df, prompts


def generate_texts(
    prompts: List[List[Dict[str, str]]],
    llm_name: str,
    llm_path: str,
    quant: str,
    sampling_params: List[SamplingParams],
    batch_size: int,
    base_path: str,
) -> None:
    if llm_name == "gpt-4.1-nano-2025-04-14":
        client = OpenAI(api_key=API_KEY)
        csv_path = f"{base_path}{llm_name}.csv"
    else:
        model = LLM(
            model=llm_path,
            quantization=quant,
            max_model_len=(
                MAX_MODEL_LEN // 2 if llm_name == "microsoft/phi-4" else MAX_MODEL_LEN
            ),
            trust_remote_code=True,
            seed=SEED,
            tensor_parallel_size=2,
        )
        csv_path = f"{base_path}{llm_name.split('/')[-1]}.csv"

    # init csv file
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "text", "temperature", "top_p", "top_k"])

    batches = list(batchify(prompts, batch_size))
    print(f"Generating texts for {llm_name}...")
    for prompts_batch in tqdm(batches, total=len(prompts) // batch_size):
        params = random.choice(sampling_params)

        if llm_name == "gpt-4.1-nano-2025-04-14":
            responses = generate_responses_gpt(
                client,
                prompts_batch,
                params.temperature,
                params.top_p,
            )
        else:
            responses = generate_responses(model, prompts_batch, params)

        save_to_csv(
            csv_path,
            prompts_batch,
            responses,
            params.temperature,
            params.top_p,
            params.top_k,
        )

    df = pd.read_csv(csv_path)
    print(
        f"Expected samples: {len(prompts)}, Actual samples: {len(df)}, Match: {len(prompts) == len(df)}, Model: {llm_name}"
    )
