from collections.abc import Generator, Iterable
import functools
import itertools
from typing import Any, Callable
import pandas as pd
import requests
import dotenv
import os
import json
import paper_dataset
import tqdm
import re
from llm_pipeline import _pipeline
import datasets
from llm_prompts import *
import pycountry
import functools

dotenv.load_dotenv()

backends = ["vllm", "hf", "api"]
def _check_backend(backend: str) -> None:
    if backend not in backends:
        raise ValueError(f"Invalid backend: {backend}. Choose from {backends}.")

def extract_languages(output: str) -> list[str]:
    parenthesis = r"\([^\(\)]{4,}\)" # match parentheses with at least 4 characters inside
    parenthesis_codes = r"\([^\(\)]{2,3}\)" # match parentheses with at least 4 characters inside
    number_list = r"^[0-9]+\. "
    forbidden = ["implied", "explicit", "no specif", "mention"]

    output = output.lstrip("*").strip()
    lines = output.split("\n")
    lines = [line for line in lines if all(f not in line.lower() for f in forbidden)]

    lines = [re.sub(parenthesis, "", line) for line in lines]  # remove parentheses with explanation
    if len(lines) == 1 or (len(lines) >= 1 and lines[0].count(",") > 3):
        lines = [line.split(",") for line in lines]
        lines = list(itertools.chain.from_iterable(lines))

    # lines = [line.split(",") for line in lines]    
    lines = [line.lstrip("*").strip() for line in lines]
    lines = [re.sub(number_list, "", line).strip() for line in lines]
    lines = [line for line in lines if line != ""]
    lines = [line for line in lines if "no specific" not in line.lower() and line.lower() != "none"]

    # find language codes, 2 or 3 letters with the same case
    langs = []
    for line in lines:
        if (line.islower() or line.isupper()) and line.isalpha():
            if len(line) == 2:
                lang = pycountry.languages.get(alpha_2=line)
            elif len(line) == 3:
                lang = pycountry.languages.get(alpha_3=line)
            else:
                lang = None
                langs.append(line) # Programming languages
            if lang is not None:
                # Remove parenthesis from name
                lang = re.sub(parenthesis, "", lang.name).strip()
                langs.append(lang)
            elif line == "tib": # Missing in pycountry
                langs.append("Tibetan")
            elif line == "eml": # Deprecated
                langs.append("Emiliano-Romagnolo")
            else:
                print(f"Warning: language code {line} not found")
    
        else:
            langs.append(line)
    # Remove codes in parentheses
    langs = [re.sub(parenthesis_codes, "", lang).strip() for lang in langs]
    langs = sorted(list(set(langs)))
    return langs

def extract_metrics(output: str) -> list[str]:
    output = output.lower().strip()
    output = re.sub(
        r"\([^\(\)]*\) ?", "", output
    ).strip()  # remove parentheses with explanation
    output = re.sub(r"\.+$", "", output)  # remove trailing dot

    if "," in output:
        output = output.split(",")
    elif "\n" in output:
        output = output.split("\n")
    else:
        output = [output]
    items = [i.strip() for i in output]
    items = [i.replace("and ", "") for i in items]  # remove leading and
    items = [i.replace("* ", "") for i in items]  # remore bullets
    items = [i for i in items if i != ""]

    metrics = sorted(list(set(items)))

    return metrics

def extract_tasks_metrics(output: str) -> list[str]:
    output = output.lower().strip()
    items = output.split("\n") # task: metric, metric
    tasks = {}
    for item in items:
        if "not explicitly mentioned" in item or "not mentioned" in item:
            continue
        item = item.lstrip("*").strip()
        if len(item) == 0:
            continue
        if ":" not in item:
            continue
        task, metrics = item.split(":", maxsplit=1)
        task = task.strip()
        task = _task_normalization(task)
        metrics = metrics.strip()
        # if "," in metrics:
        #     metrics = metrics.split(",")
        # else:
        #     metrics = metrics.split(" ")
        metrics = metrics.split(",")
        metrics = [m.strip() for m in metrics]
        metrics = [_metric_normalization(m) for m in metrics]
        metrics = list(itertools.chain.from_iterable(metrics))
        metrics = list(set([m for m in metrics if m != ""]))
        tasks[task] = metrics
    return tasks
    
def _metric_normalization(metric: str) -> str:
    if metric.startswith("recall@"):
        return ["recall@k"]
    if metric.startswith("precision@"):
        return ["precision@k"]
    if "accuracy" in metric and ("top-" in metric or "@" in metric):
        return ["accuracy@k"]
    elif "accuracy" in metric:
        return ["accuracy"]
    if "precision" in metric and ("top-" in metric or "@" in metric):
        return ["precision@k"]
    elif "precision" in metric:
        return ["precision"]
    if "recall" in metric and ("top-" in metric or "@" in metric):
        return ["recall@k"]
    elif "recall" in metric:
        return ["recall"]
    if "f1" in metric:
        return ["f1 score"]
    if "perplexity" in metric:
        return ["perplexity"]
    if "speed up" in metric or "speed-up" in metric or "speedup" in metric:
        return ["speedup"]
    if "latency" in metric:
        return ["latency"]
    if "flops" in metric:
        return ["flops"]
    if "correlation" in metric:
        return ["correlation"]
    
    mapping = {
        "character error rate (cer)": ["character error rate"],
        "word error rate (wer)": ["word error rate"],
        "cer": ["character error rate"],
        "wer": ["word error rate"],
        "ppl": ["perplexity"],
        "bleu chrf comet": ["bleu", "chrf", "comet"],
        "bleu comet": ["bleu", "comet"],
        "ppl_best": ["perplexity"],
        "entropy value": ["entropy"],
        "exact match (em)": ["exact match"],
        "f-score": ["f1 score"], # we assume it's f1
        "f value": ["f1 score"],
        "bleu lcs f-score cer acc": ["bleu", "lcs", "f1 score", "character error rate", "accuracy"],
        "gflops": ["flops"],
        "bleu score": ["bleu"],        
    }

    tmp = mapping.get(metric, None)
    if tmp is not None:
        return tmp
    
    if metric.startswith("rouge"):
        return ["rouge"]
    if metric.startswith("bleu") and not metric.startswith("bleurt"):
        return ["bleu"]
    if metric.startswith("chrf"): # chrf, chrf++, chr2
        return ["chrf"]
    return [metric]

def _task_normalization(task: str) -> str:
    # Iteratively add synonyms
    if "embeddings" in task:
        task = task.replace("embeddings", "embedding")
    if "modelling" in task:
        task = task.replace("modelling", "modeling")
    if "summarisation" in task:
        task = task.replace("summarisation", "summarization")
    if "morphological" in task:
        task = task.replace("morphological", "morpheme")
    if "tokenisation" in task:
        task = task.replace("tokenisation", "tokenization")
    if "neural machine translation" in task:
        task = task.replace("neural machine translation", "machine translation")
    if "translation" in task:
        task = "machine translation"
    if "decoding" in task:
        task = "decoding"
    if task in ["subwording", "segmentation", "subword segmentation", "text segmentation", "word segmentation"]:
        task = "tokenization"
    if task == "pos tagging":
        task = "part-of-speech tagging"
    if "named entity recognition" in task or "(ner)" in task:
        task = "named entity recognition"
    if "model comparison" in task or task in ["evaluation", "error analysis", "model performance", "downstream tasks", "comparison", "model analysis", "analysis", "robustness evaluation"]:
        task = "model evaluation"
    if task in ["model efficiency", "efficiency", "computational efficiency"]:
        task = "model efficiency"
    if task in ["language modeling", "masked language modeling"]:
        task = "language modeling"
    if task in ["text summarization", "summarization"]:
        task = "summarization"
    return task

def extract_units(output: str) -> list[str]:
    output = output.lower().strip()
    output = re.sub(
        r"\([^\(\)]*\) ?", "", output
    ).strip()  # remove parentheses with explanation
    output = re.sub(r"\.+$", "", output)  # remove trailing dot

    if "," in output:
        output = output.split(",")
    elif "\n" in output:
        output = output.split("\n")
    else:
        output = [output]
    items = [i.strip() for i in output]
    items = [i.replace("and ", "") for i in items]  # remove leading and
    items = [i.replace("* ", "") for i in items]  # remore bullets
    items = [i.replace("sub-word", "subword") for i in items]  # normalize sub-word
    items = [i for i in items if i != ""]

    tmp = []
    for item in items: # handcrafted normalization
        if item in ["token", "tokens", "subword tokens", "subword", "subword units", "semantic tokens", "text tokens", "segments", "wordpieces"]:
            tmp.append("subwords")
        elif item == "unicode characters":
            tmp.append("characters")
        elif item == "character":
            tmp.append("characters")
        elif item in ["morpheme", "morphemes","morphs"]:
            tmp.append("morphemes")
        elif item in ["phoneme", "phonemes", "phones"]:
            tmp.append("phonemes")
        elif "\n" in item or ")" in item:
            pass
        else:
            tmp.append(item)

    units = sorted(list(set(tmp)))

    return units


def judge(
    system_prompt: str,
    user_prompt: str,
    items: dict[str, str],
    max_tokens: int = 1,
    backend: str = "hf",
) -> str:
    _check_backend(backend)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt.format(**items)},
    ]
    if backend == "vllm":
        model_name = os.environ["LOCAL_LLM_MODEL"]
        answer = _judge_vllm(model_name, messages, max_new_tokens=max_tokens)
    elif backend == "hf":
        model_name = os.environ["LOCAL_LLM_MODEL"]
        answer = _judge_hf(model_name, messages, max_new_tokens=max_tokens)
    else:
        model_name = os.environ["LLM_MODEL"]
        answer = _judge_api(model_name, messages, max_new_tokens=max_tokens)
    return answer.strip()


def _judge_vllm(
    model_name: str, messages: list[dict[str, str]], max_new_tokens: int = 1
) -> str:
    model = _pipeline(model_name)
    sampling_params = model.get_default_sampling_params()
    sampling_params.max_tokens = max_new_tokens
    answer = (
        model.chat(messages, use_tqdm=False, sampling_params=sampling_params)[-1]
        .outputs[-1]
        .text
    )
    return answer


def _judge_hf(
    model_name: str, messages: list[dict[str, str]], max_new_tokens: int = 1
) -> str:
    model = _pipeline(model_name, hf=True)
    answer = model(messages, do_sample=False, max_new_tokens=max_new_tokens)[-1][
        "generated_text"
    ][-1]["content"].strip()
    return answer


def _judge_api(
    model_name: str, messages: list[dict[str, str]], max_new_tokens: int = 1
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
        answer = response.json()["choices"][0]["message"]["content"]
    else:
        raise RuntimeError(f"Response error {response.status_code}: {response.text}")
    return answer


def filter_topic(
    abstracts: list[str], titles: list[str], backend: str = "hf", tqdm_disable: bool = False
) -> bool | Generator[bool, None, None]:
    def items(titles, abstracts):
        for t, a in zip(titles, abstracts):
            yield {"title": t, "abstract": a}
    gen = judge_generator(
        SYSTEM_PROMPT_FILTER, 
        TOKENIZATION_EXAMPLES + USER_PROMPT_TITLE_ABSTRACT_YN,
        items(titles, abstracts),
        backend=backend
    )
    for answers in tqdm.tqdm(gen, disable=tqdm_disable, total=len(titles), desc="Filtering"):
        yield answers.lower().startswith("yes")
    


def is_tokeniser(
    abstract: str, title: str, backend: str = "hf"
) -> bool:
    items = {"title": title, "abstract": abstract}
    answer = judge(
        SYSTEM_PROMPT_IS_TOKENISER,
        TOKENIZATION_EXAMPLES + USER_PROMPT_TITLE_ABSTRACT_YN,
        items,
        backend=backend,
    )
    return answer.lower().startswith("yes")


def is_evaluation(
    abstract: str, title: str, backend: str = "hf"
) -> bool:
    items = {"title": title, "abstract": abstract}
    answer = judge(
        SYSTEM_PROMPT_IS_EVALUATION,
        USER_PROMPT_TITLE_ABSTRACT_YN,
        items,
        backend=backend,
    )
    return answer.lower().startswith("yes")


def is_survey(
    abstract: str, title: str, backend: str = "hf"
) -> bool:
    items = {"title": title, "abstract": abstract}
    answer = judge(
        SYSTEM_PROMPT_IS_SURVEY,
        USER_PROMPT_TITLE_ABSTRACT_YN,
        items,
        backend=backend,
    )
    return answer.lower().startswith("yes")


def judge_generator(
    system_prompt: str,
    user_prompt: str,
    items_ds: Iterable[dict[str, str]],
    max_tokens: int = 1,
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
        model = _pipeline(model_name, hf=True)
        for answer in model(fmt(items_ds), do_sample=False, max_new_tokens=max_tokens):
            yield answer[-1]["generated_text"][-1]["content"].strip()
    elif backend == "vllm":
        model_name = os.environ["LOCAL_LLM_MODEL"]
        judge_fn = functools.partial(
            _judge_vllm, model_name=model_name, max_new_tokens=max_tokens
        )
        model = _pipeline(model_name)
        sampling_params = model.get_default_sampling_params()
        sampling_params.max_tokens = max_tokens
        for answer in model.chat(fmt(items_ds), use_tqdm=False, sampling_params=sampling_params):
            yield answer.outputs[-1].text.strip()     
    else:
        model_name = os.environ["LLM_MODEL"]
        judge_fn = functools.partial(
            _judge_api, model_name=model_name, max_new_tokens=max_tokens
        )
        for messages in fmt(items_ds):
            answer = judge_fn(messages=messages).strip()
            yield answer


def _get_features(
    items: pd.DataFrame,
    backend: str = "hf",
    system_prompt: str = "",
    user_prompt: str = "",
    remove_sections: bool = False,
    desc: str = "",
    disable: bool = False,
    file: str = os.devnull,
) -> Generator[str, None, None]:
    def ds_gen():
        for item in items.iloc:
            if item["exclude"]:
                yield {"title": "", "content": ""}
            else:
                md = paper_dataset.get_markdown(item["pdf_url"])
                if remove_sections:
                    md = paper_dataset.split_sections(md)
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


def get_metrics(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_METRICS,
        user_prompt=USER_PROMPT_METRICS,
        desc="Metrics",
        file=file,
    )

def get_metrics_tasks(
    items, backend: str = "hf", file: str = os.devnull
) -> Generator[str, None, None]:
    return _get_features(
        items,
        backend=backend,
        system_prompt=SYSTEM_PROMPT_METRICS_TASKS,
        user_prompt=USER_PROMPT_METRICS_TASKS,
        desc="Metrics + Tasks",
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


def test(file: str, fn: Callable[[str, str, str], Any], answer_key) -> None:
    samples = []
    with open(file, "r") as f:
        samples = json.load(f)
    answer = []
    for sample in tqdm.tqdm(samples):
        content = sample.get("content", None)
        if content is None or content == "":
            content = ""
        else:
            content = paper_dataset.to_markdown(content)
        answer.append(fn(sample.get("abstract"), sample.get("title"), content))
    ref = [sample.get(answer_key) for sample in samples]
    for s, a, r in zip(samples, answer, ref):
        print(f"Title: {s.get('title')}")
        print(f"Answer: {a} (Reference: {r})")
        print("-" * 80)


def has_pypi(md: str) -> bool:
    pattern = r"\[.*?\]\(https://pypi\.org/project/.*?\)"
    return re.search(pattern, md) is not None


def has_repository(md: str) -> bool:
    # github, bitbucket, gitlab, ...
    github = r"\[.*?\]\(https://github\.com/.*?\)"
    bitbucket = r"\[.*?\]\(https://bitbucket\.org/.*?\)"
    gitlab = r"\[.*?\]\(https://gitlab\.com/.*?\)"
    pattern = f"({github})|({bitbucket})|({gitlab})"
    return re.search(pattern, md) is not None
