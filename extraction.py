import re
import itertools
import pycountry
import functools
from extraction_maps import *
from typing import Callable





def extract_languages(output: str) -> list[str]:
    parenthesis = (
        r"\([^\(\)]{4,}\)"  # match parentheses with at least 4 characters inside
    )
    parenthesis_codes = (
        r"\([^\(\)]{2,3}\)"  # match parentheses with at least 4 characters inside
    )
    number_list = r"^[0-9]+\. "
    empty = ["not specified", "none", "..."]
        
    lines = output.split("\n")
    lines = [line.lstrip("*").strip() for line in lines]
    lines = [line for line in lines if line.lower() not in empty]
    lines = [line for line in lines if "duplicate" not in line.lower()]
    lines = [line.replace("(inferred)", "").strip() for line in lines]

    lines = [
        re.sub(parenthesis, "", line) for line in lines
    ]  # remove parentheses with explanation    
        
    lines = [re.sub(number_list, "", line).strip() for line in lines]
    lines = [line for line in lines if line != ""]
    
    # find language codes, 2 or 3 letters with the same case
    langs = []
    for line in lines:
        line = line.split("_")[0].strip() # Remove script code
        line = line.split("–")[0].strip() # Remove explanation after dash
        if (line.islower() or line.isupper()) and line.isalpha():            
            if len(line) == 2:
                lang = pycountry.languages.get(alpha_2=line)
            elif len(line) == 3:
                lang = pycountry.languages.get(alpha_3=line)
            else:
                lang = None
                langs.append(line)  # Programming languages
            if lang is not None:
                # Remove parenthesis from name
                lang = re.sub(parenthesis, "", lang.name).strip()
                langs.append(lang)
            elif line == "tib":  # Missing in pycountry
                langs.append("Tibetan")
            elif line == "eml":  # Deprecated
                langs.append("Emiliano-Romagnolo")
            else:
                print(f"Warning: language code {line} not found")
        else:
            langs.append(line)
    # Remove codes in parentheses
    langs = [re.sub(parenthesis_codes, "", lang).strip() for lang in langs]
    langs = sorted(list(set(langs)))
    return langs


def extract_units(output: str, unit_mapping: dict[str, list[str]] = None, other: bool = False) -> list[str]:
    output = output.lower().strip()
    output = re.sub(r"\.+$", "", output)  # remove trailing dot

    if "\n" in output:
        output = output.split("\n")
    else:
        output = [output]
    items = [i.strip() for i in output]
    items = [i.replace("and ", "") for i in items]  # remove leading and
    items = [i.replace("* ", "") for i in items]  # remove bullets
    items = [i for i in items if i != ""]

    if unit_mapping is not None:
        inv_map = invert_mapping(unit_mapping)
        if other:
            items = [inv_map.get(item, "other") for item in items]
        else:
            items = [inv_map.get(item, item) for item in items]
    units = sorted(list(set(items)))
    return units

def has_evaluation(output: str) -> bool:
    output = output.lower()
    return output != "no" 

def extract_tasks(output: str, task_mapping: Callable[[str], str] | None = None, other: bool = False) -> list[str]:
    output = output.lower().strip()
    output = re.sub(
        r"\([^\(\)]*\) ?", "", output
    ).strip()  # remove parentheses with explanation
    output = re.sub(r"\.+$", "", output)  # remove trailing dot
    
    output = output.replace("‑", "-") # normalize dashes
    output = output.replace("\u202f", " ")
    output = output.split("\n")
    tasks = [i.strip() for i in output]
    tasks = [i.replace("* ", "") for i in tasks]  # remove bullets
    tasks = [i for i in tasks if i != ""]
    tasks = [i for i in tasks if i != "no"]

    if task_mapping is not None:        
        if other:
            tasks = [task_mapping(task) if task_mapping(task) is not None else "other" for task in tasks]
        else:
            tasks = [task_mapping(task) if task_mapping(task) is not None else task for task in tasks]

    return tasks

def extract_metrics(output: str, metric_mapping: Callable[[str], str] | None = None, other: bool = False) -> list[str]:
    output = output.lower().strip()
    output = re.sub(
        r"\([^\(\)]*\) ?", "", output
    ).strip()  # remove parentheses with explanation
    output = re.sub(r"\.+$", "", output)  # remove trailing dot
    
    output = output.replace("‑", " ") # different utf characters
    output = output.replace("-", " ")
    output = output.replace("\u202f", " ")
    output = output.replace("₁", "1")
    output = output.replace("’", "")  # remove apostrophes
    output = output.replace("out of vocabulary", "oov")  # normalize oov
    output = output.split("\n")

    # Remove top-[0-9]+, @[0-9]+, number at the end from metrics
    output = [re.sub(r"top ?[0-9]+ ?", "", metric).strip() for metric in output]
    output = [re.sub(r"@ ?[0-9]+ ?", "", metric).strip() for metric in output]
    output = [re.sub(r"\b[0-9]+$", "", metric).strip() for metric in output] 
    # Remove mean, average, macro, micro, normalized, overall from metrics
    words = ["mean", "average", "averaged", "macro", "micro", "normalized", "overall", "full", "avg", "exact", "perfect", "proportion of", "number of", "and", "percentage of", "pos tagging", "part of speech tagging", "ner", "pos", ]
    for word in words:
        output = [re.sub(rf"\b{word}\b", "", metric).strip() for metric in output]

    # Normalize whitespace
    output = [re.sub(r"\s+", " ", metric).strip() for metric in output]
    metrics = [i.strip() for i in output]
    metrics = [i.replace("* ", "").strip() for i in metrics]  # remove bullets
    metrics = [i for i in metrics if i != ""]
    metrics = [i for i in metrics if i != "no"]

    if metric_mapping is not None:
        inv_map = invert_mapping(metric_mapping)
        if other:
            metrics = [inv_map.get(metric, "other") for metric in metrics]
        else:
            metrics = [inv_map.get(metric, metric) for metric in metrics]
    return metrics