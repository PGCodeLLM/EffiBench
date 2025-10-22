import json
import openai
import argparse
import os
from tqdm import tqdm
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setting API parameters
# BEN: Commenting this out as we override via env vars
# openai.api_key = 'API'

with open("../prompts/prompt.txt", "r") as f:
    text = f.read()

# Function to fetch completion
def fetch_completion(data_entry, model):
    global text
    test_case = data_entry["small_test_cases"]
    try:
        completions = openai.ChatCompletion.create(
            model=model,
            stream=False,
            messages=[
                {"role": "system", "content": "You are a code developer."},
                {
                    "role": "user",
                    "content": (
                        f"{text}\n"
                        f"# Task description:\n```python\n{data_entry['markdown_description']}\n```\n"
                        f"# Test case:\n```python\n{test_case}\n```"
                    )
                },
            ],
            request_timeout=100,
        )
        data_entry["completion"] = completions.choices[0]["message"]["content"]
    except Exception as e:
        print(f"Error processing entry '{data_entry.get('id', 'unknown')}': {e}")
        data_entry["completion"] = ""

    return data_entry


# BEN: This has been updated to accept optional args for evalhub integration
def fetch_completion_with_args(data_entry, model, args=None):
    global text
    test_case = data_entry["small_test_cases"]
    try:
        create_kwargs = {
            "model": model,
            "stream": getattr(args, "stream", False),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code developer."
                },
                {
                    "role": "user",
                    "content": (
                        f"{text}\n"
                        f"# Task description:\n```python\n{data_entry['markdown_description']}\n```\n"
                        f"# Test case:\n```python\n{test_case}\n```"
                    )
                },
            ],
            "request_timeout": 100,
        }
        # Add optional parameters if set
        if args is not None:
            if args.temperature is not None:
                create_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                create_kwargs["top_p"] = args.top_p
            if args.top_k:
                create_kwargs["top_k"] = args.top_k
            if args.repetition_penalty is not None:
                create_kwargs["repetition_penalty"] = args.repetition_penalty
            if args.presence_penalty is not None:
                create_kwargs["presence_penalty"] = args.presence_penalty
            if args.n and args.n > 1:
                create_kwargs["n"] = args.n
            if args.max_tokens and args.max_tokens > 0:
                create_kwargs["max_tokens"] = args.max_tokens
        completions = openai.ChatCompletion.create(**create_kwargs)
        data_entry["completion"] = completions.choices[0]["message"]["content"]
    except Exception as e:
        print(f"Error processing entry '{data_entry.get('id', 'unknown')}': {e}")
        data_entry["completion"] = ""

    return data_entry


def add_custom_arguments(parser):
    # BEN: I dont know if all of these will be necessary, but adding them for completeness
    parser.add_argument('--exp-id', type=str, default='', help='Experiment ID')
    parser.add_argument('--model-v1-endpoint', type=str, default='', help='Model v1 endpoint')
    parser.add_argument('--api-key', type=str, default='', help='API key')
    parser.add_argument('--num-workers', type=int, default=10, help='Number of workers')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=1.0, help='Nucleus sampling probability')
    parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty')
    parser.add_argument('--presence-penalty', type=float, default=0.0, help='Presence penalty')
    parser.add_argument('--n', type=int, default=1, help='Number of completions to generate')
    parser.add_argument('--max-tokens', type=int, default=0, help='Maximum number of tokens')
    parser.add_argument('--extra-body', type=str, default='', help='Extra body for API call')
    parser.add_argument('--extra-headers', type=str, default='', help='Extra headers for API call')
    parser.add_argument('--stream', action='store_true', help='Enable streaming responses')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch completions using OpenAI ChatCompletion API.')
    parser.add_argument('--model', '-m', type=str, default='gpt-3.5-turbo', help='Model to use for completion')
    add_custom_arguments(parser)
    args = parser.parse_args()
    model = args.model

    with open("../data/dataset.json", "r") as f:
        dataset = json.load(f)

    with ThreadPoolExecutor(max_workers=10) as executor:
        # BEN: We will call fetch_completion_with_args instead of fetch_completion
        future_to_entry = {
            executor.submit(fetch_completion_with_args, copy.deepcopy(entry), model, args): entry
            for entry in tqdm(dataset, desc="Submitting tasks")
        }
        for future in tqdm(as_completed(future_to_entry), total=len(future_to_entry), desc="Processing tasks"):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(f"Error updating entry '{entry.get('id', 'unknown')}': {e}")

    # Ensure the results directory exists
    os.makedirs("../results", exist_ok=True)
    # BEN: We'll use the experiment ID rather than the model for all output directories
    # output_file = f"../results/{model.replace('/', '_')}.json"
    output_file = f"../results/{args.exp_id}.json"

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"Results saved to {output_file}")
