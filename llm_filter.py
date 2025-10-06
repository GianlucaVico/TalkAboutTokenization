from typing import Any, Callable
import requests
import dotenv
import os
# from transformers import pipeline
from vllm import LLM, SamplingParams
import json
import paper_dataset
import tqdm
import re
dotenv.load_dotenv()

USER_PROMPT_TITLE_ABSTRACT_YN = (
    "The title is: '{title}'. "
    "The abstract is: '{abstract}'. "
    "Answer 'Yes' or 'No' and nothing else."
)
USER_PROMPT_TITLE_CONTENT_YN = (
    "The title is: '{title}'. "
    "The abstract is: '{content}'. "
    "Answer 'Yes' or 'No' and nothing else."
)
USER_PROMPT_TITLE_CONTENT_OPEN = ( # TODO test
    "The title is: '{title}'. "
    "The content is: '{content}'. "
    "The answer is: "
)



SYSTEM_PROMPT_FILTER = ( # ok for pixels, dynamic chunking, tok and noiseless channel, tokenization cost
    "You are a reviewer tasked with determining whether a research paper is related to tokenization based on its title and abstract. "
    "Given the title and abstract of the paper, decide if it is primarily focused on tokenization or not. "
    "This includes tokenization, tokenisation, subword segmentation, token free, vocabulary methods, alternatives to tokens."
)

SYSTEM_PROMPT_IS_TOKENISER = ( 
    "You are a reviewer tasked with determining whether a research paper proposes a novel tokenization method based on its title and abstract. "
    "Given the title and abstract of the paper, decide if its main goal is to introduce a new approach, technique, or algorithm for tokenizing text."
)

# SYSTEM_PROMPT_IS_APPLICATION = ( # TODO test
#     "You are a reviewer tasked with determining whether a research paper applies tokenization to a specific application "
#     "or task in natural language processing (NLP) or related areas, beyond the typical use cases. "
#     "Given the title and abstract of the paper, decide if it explores the use of tokenization in various NLP tasks or applications."
# )


SYSTEM_PROMPT_IS_EVALUATION = ( # ok for myte, noiseless channel, cost
    # "You are a reviewer tasked with determining whether the topic of a research paper is metrics for tokenizers. "
    # "Given the title and abstract of the paper, decide if its main goal is to propose nover evaluation metrics to "
    # "assess the performance, effectiveness, or quality of tokenization techniques."
    # "You are a reviewer tasked with determining whether the research paper is about evaluating tokenizers rather than proposing new tokenization methods. "
    # "Given the title and abstract of the paper, decide if its main goal is to propose nover evaluation metrics to assess the performance, effectiveness, "
    # "or quality of tokenization techniques."
    "You are a reviewer tasked with determining whether the main topic of a research paper is evaluating tokenizers rather than proposing new tokenization methods. "
    "Given the title and abstract of the paper, decide if its main goal is to discuss evaluation, performance, effectiveness, or quality of tokenization techniques."

)

SYSTEM_PROMPT_IS_SURVEY = ( 
    "You are a reviewer tasked with identifying research papers that provide a comprehensive overview or survey of tokenization methods, "
    "techniques, and approaches in natural language processing. Given the title and abstract of the paper, decide if it presents a "
    "systematic review, comparison, or analysis of existing tokenization methodologies, or provides a taxonomy, framework, or landscape of tokenization research."
)

SYSTEM_PROMPT_UNIT = ( # TODO test
    "You are a reviewer tasked with identifying the basic unit of analysis used by a tokenizer based on the title and content of a research paper. "
    "Given the title and content of the paper, determine what smaller units (such as words, characters, bytes, subwords, morphemes, pixels, or other) "
    "the tokenizer operates on. Answer only with the most appropriate unit and nothing else."
)

SYSTEM_PROMPT_LANGUAGE = ( 
    "You are a reviewer tasked with identifying the languages that are used as test cases, evaluation datasets, or experimental targets "
    "in a research paper on tokenization given the title and content of the paper. "
)
USER_PROMPT_LANGUAGE_EN = ( # Output Yes/No
    "The title is: '{title}'. \n"
    "The content is '{content}'. \n\n"
    "Does the paper use exclusively English as the language for experiments, datasets, or evaluation? Answer 'Yes' or 'No' and nothing else. "
)
# Output:
#Here is the list of languages:

# 1. English (en)
# 2. Vietnamese (vi)
# 3. Chinese (zh)
# 4. French (fr)
# 5. Arabic (ar)
# 6. Japanese (ja)
# 7. German (de)
# 8. Czech (cs)
USER_PROMPT_LANGUAGE_MULTILINGUAL = (
    "The title is: '{title}'. \n"
    "The content is: '{content}'. \n\n"
    "List all the languages mentioned in the paper that are used as test cases, evaluation datasets, or experimental targets and nothing else: "
) 
 
def extract_languages(output: str) -> list[str]:
    lines = output.lower().strip().split("\n")
    lines = [line.strip() for line in lines if line.strip() != '']    
    # 1. English (en) -> group 1: English (en)
    r = r"(?:[0-9]+\.|\*) (?P<l1>[\w \(\)'\.\-,]+)|(?P<l2>english)|(?P<l3>chinese)" 
    matches = [re.search(r, line) for line in lines]
    matches = [m for m in matches if m is not None]
    # languages = [m.group(1) for m in matches if m.group(1) is not None]
    languages = []
    for g in ['l1', 'l2', 'l3']:
        languages.extend([m.group(g) for m in matches if m.group(g) is not None])
    languages = [lang.strip().lower() for lang in languages]
    languages = [lang for lang in languages if lang != '']
    languages = list(set(languages))
    return languages


class _pipeline:
    def __new__(cls, model_name: str):
        if not hasattr(cls, 'model'):
            # cls.model = pipeline("text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
            cls.model = LLM(model_name, tensor_parallel_size=4, dtype='auto')
        return cls.model


def judge(system_prompt: str, user_prompt: str, items: dict[str, str], max_tokens: int = 1, local: bool = False) -> str:
    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt.format(**items)}
    ]
    if local:
        model_name = os.environ["LOCAL_LLM_MODEL"]
        answer = _judge_local(model_name, messages, max_new_tokens=max_tokens)
    else:
        model_name = os.environ["LLM_MODEL"]
        answer = _judge_api(model_name, messages, max_new_tokens=max_tokens)
    return answer.strip()
    

def _judge_local(model_name: str, messages: list[dict[str, str]], max_new_tokens: int = 1) -> str:
    # model = _pipeline('text-generation', model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    model = _pipeline(model_name)
    sampling_params = model.get_default_sampling_params()
    sampling_params.max_tokens = max_new_tokens
    # result = model(messages, max_new_tokens=max_new_tokens)[0]
    # response = result[0]["generated_text"]
    # answer = next(filter(lambda x: x["role"] == "assistant", response))["content"]
    answer = model.chat(messages, use_tqdm=False, sampling_params=sampling_params)[-1].outputs[-1].text
    return answer

def _judge_api(model_name: str, messages: list[dict[str, str]], max_new_tokens: int = 1) -> str:
    headers = {
        'Authorization': f'Bearer {os.environ["LLM_TOKEN"]}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": model_name,
      "messages": messages,
      "max_tokens": max_new_tokens, 
      "timeout": 600,
    }
    answer = ""
    response = requests.post(os.environ["LLM_URL"], headers=headers, json=data, timeout=600)
    if response.status_code == 200:
        answer = response.json()['choices'][0]['message']['content']        
    else:
        raise RuntimeError(f"Response error {response.status_code}: {response.text}")
    return answer

def filter_topic(abstract: str, title: str) -> bool:
    items = {
        "title": title,
        "abstract": abstract
    }
    answer = judge(SYSTEM_PROMPT_FILTER, USER_PROMPT_TITLE_ABSTRACT_YN, items)
    return answer.lower().startswith("yes")

def is_tokeniser(abstract: str, title: str) -> bool:
    items = {
        "title": title,
        "abstract": abstract
    }
    answer = judge(SYSTEM_PROMPT_IS_TOKENISER, USER_PROMPT_TITLE_ABSTRACT_YN, items)
    return answer.lower().startswith("yes")

def is_evaluation(abstract: str, title: str) -> bool:
    items = {
        "title": title,
        "abstract": abstract
    }
    answer = judge(SYSTEM_PROMPT_IS_EVALUATION, USER_PROMPT_TITLE_ABSTRACT_YN, items)
    return answer.lower().startswith("yes")

def is_survey(abstract: str, title: str) -> bool:
    items = {
        "title": title,
        "abstract": abstract
    }
    answer = judge(SYSTEM_PROMPT_IS_SURVEY, USER_PROMPT_TITLE_ABSTRACT_YN, items)
    return answer.lower().startswith("yes")

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
    pattern = r'\[.*?\]\(https://pypi\.org/project/.*?\)'
    return re.search(pattern, md) is not None

def has_repository(md: str) -> bool:
    # github, bitbucket, gitlab, ...
    github = r'\[.*?\]\(https://github\.com/.*?\)'
    bitbucket = r'\[.*?\]\(https://bitbucket\.org/.*?\)'
    gitlab = r'\[.*?\]\(https://gitlab\.com/.*?\)'
    pattern = f'({github})|({bitbucket})|({gitlab})'
    return re.search(pattern, md) is not None