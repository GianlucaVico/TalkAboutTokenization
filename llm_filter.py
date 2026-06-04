from collections.abc import Generator, Iterable
import pandas as pd
import requests
import dotenv
import os
import paper_dataset
import tqdm
import re
from llm_pipeline import _pipeline
import datasets
from llm_prompts import *
import time
import sys
import functools

dotenv.load_dotenv()

RE_FILTER = re.compile(r"token|sub-?word|segment|sub-?tok")

backends = ["hf", "api"]
def _check_backend(backend: str) -> None:
    if backend not in backends:
        raise ValueError(f"Invalid backend: {backend}. Choose from {backends}.")

def _judge_api(
    model_name: str, messages: list[dict[str, str]], max_new_tokens: int = 1024
) -> str:
    headers = {
        "Authorization": f'Bearer {os.environ["LLM_TOKEN"]}',
        "Content-Type": "application/json",
    }
    data = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "timeout": 600,
    }
    answer = ""
    response = requests.post(
        os.environ["LLM_URL"], headers=headers, json=data, timeout=600
    )
    if response.status_code == 200:
        try:
            answer = response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            # raise RuntimeError(f"Response parsing error: {e}. Response text: {response.text}")
            print(f"\nResponse parsing error: {e}. Response text: {response}\n", file=sys.stderr)
            answer = ""
    else:
        # raise RuntimeError(f"Response error {response.status_code}: {response.text}")
        print(f"\nResponse error {response.status_code}: {response}\n", file=sys.stderr)
        answer = ""
    return answer

def judge_generator(
    system_prompt: str,
    user_prompt: str,
    items_ds: Iterable[dict[str, str]],
    max_tokens: int = 1024,
    backend: str = "hf",
) -> Generator[str, None, None]:
    _check_backend(backend)
    def fmt(item_ds):
        for items in item_ds:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt.format(**items)},
            ]
            yield messages

    if backend == "hf":
        model_name = os.environ["LOCAL_LLM_MODEL"]
        model = _pipeline(model_name)
        for answer in model(fmt(items_ds), do_sample=False, max_new_tokens=max_tokens):
            tmp = answer[-1]["generated_text"][-1]["content"]
            yield tmp.split("assistantfinal")[-1].strip()      
    else:
        model_name = os.environ["LLM_MODEL"]
        judge_fn = functools.partial(
            _judge_api, model_name=model_name, max_new_tokens=max_tokens
        )
        for messages in fmt(items_ds):
            answer = judge_fn(messages=messages).strip()
            time.sleep(10)
            yield answer

def filter_topic(
    abstracts: list[str], titles: list[str], backend: str = "hf", zero_shot: bool = False, tqdm_disable: bool = False, file: str = os.devnull
) -> Generator[bool, None, None]:
    def items(titles, abstracts):
        for t, a in zip(titles, abstracts):
            yield {"title": t, "abstract": a}
    user_prompt = USER_PROMPT_TITLE_ABSTRACT_YN if zero_shot else TOKENIZATION_EXAMPLES + USER_PROMPT_TITLE_ABSTRACT_YN
    gen = judge_generator(
        SYSTEM_PROMPT_FILTER, 
        user_prompt,
        items(titles, abstracts),
        backend=backend
    )
    with open(file, "w") as f:
        for answer in tqdm.tqdm(gen, disable=tqdm_disable, total=len(titles), desc="Filtering"):
            f.write(answer.replace("\n", "\\n") + "\n")            
            yield answer.lower().startswith("yes")

def filter_tokfree(
    abstracts: list[str], titles: list[str], backend: str = "hf", zero_shot: bool = False, tqdm_disable: bool = False, file: str = os.devnull
) -> Generator[bool, None, None]:
    def items(titles, abstracts):
        for t, a in zip(titles, abstracts):
            yield {"title": t, "abstract": a}
    user_prompt = USER_PROMPT_TITLE_ABSTRACT_YN if zero_shot else TOKENIZATION_EXAMPLES + USER_PROMPT_TITLE_ABSTRACT_YN
    gen = judge_generator(
        SYSTEM_PROMPT_FILTER_TOKFREE, 
        user_prompt,
        items(titles, abstracts),
        backend=backend
    )
    with open(file, "w") as f:
        for answer in tqdm.tqdm(gen, disable=tqdm_disable, total=len(titles), desc="Filtering"):
            f.write(answer.replace("\n", "\\n") + "\n")            
            yield answer.lower().startswith("yes")

def _get_features(
    items: pd.DataFrame,
    backend: str = "hf",
    system_prompt: str = "",
    user_prompt: str = "",
    remove_sections: bool = True,
    desc: str = "",
    disable: bool = False,
    file: str = os.devnull,
) -> Generator[str, None, None]:
    def ds_gen():
        for item in items.iloc:
            md = paper_dataset.get_markdown(item["pdf_url"])
            if remove_sections:
                md = "\n\n".join(paper_dataset.split_sections(md))
            title = item["title"] if item["title"] is not None else ""
            yield {"title": title, "content": md}

    n = len(items)
    ds = datasets.Dataset.from_generator(ds_gen)

    judge_gen = judge_generator(
        system_prompt, user_prompt, ds, backend=backend, max_tokens=1024
    )
    with open(file, "w") as f:
        for answer in tqdm.tqdm(judge_gen, total=n, desc=desc, disable=disable):
            f.write(answer.replace("\n", "\\n") + "\n")
            yield answer


def get_units(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_UNIT,
        user_prompt=USER_PROMPT_UNIT,
        desc="Units",
        file=file,
    )

def get_languages(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_LANGUAGE_MULTILINGUAL,
        user_prompt=USER_PROMPT_LANGUAGE_MULTILINGUAL,
        desc="Languages",
        file=file,
    )

def get_motivation(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_MOTIVATION,
        user_prompt=USER_PROMPT_MOTIVATION,
        desc="Motivation",
        file=file,
    )

def get_evaluation_intrinsic(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_EVALUATION_INTRINSIC,
        user_prompt=USER_PROMPT_EVALUATION_INTRINSIC,
        desc="Evaluation",
        file=file,
    )

def get_evaluation_extrinsic(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_EVALUATION_EXTRINSIC,
        user_prompt=USER_PROMPT_EVALUATION_EXTRINSIC,
        desc="Evaluation",
        file=file,
    )

def get_languages_specific(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_LANGUAGE_SPECIFIC,
        user_prompt=USER_PROMPT_LANGUAGE_LANGUAGE_SPECIFIC,
        desc="Language Specificity",
        file=file,
    )
