import re
import itertools
import pycountry
import functools
from extraction_maps import *

@functools.cache
def invert_mapping(mapping: dict[str, list[str]]) -> dict[str, str]:
    inverted = {}
    for key, values in mapping.items():
        for value in values:
            inverted[value] = key
    return inverted



def extract_languages(output: str) -> list[str]:
    parenthesis = (
        r"\([^\(\)]{4,}\)"  # match parentheses with at least 4 characters inside
    )
    parenthesis_codes = (
        r"\([^\(\)]{2,3}\)"  # match parentheses with at least 4 characters inside
    )
    number_list = r"^[0-9]+\. "
    forbidden = ["implied", "explicit", "no specif", "mention"]

    output = output.lstrip("*").strip()
    lines = output.split("\n")
    lines = [line for line in lines if all(f not in line.lower() for f in forbidden)]

    lines = [
        re.sub(parenthesis, "", line) for line in lines
    ]  # remove parentheses with explanation
    if len(lines) == 1 or (len(lines) >= 1 and lines[0].count(",") > 3):
        lines = [line.split(",") for line in lines]
        lines = list(itertools.chain.from_iterable(lines))

    # lines = [line.split(",") for line in lines]
    lines = [line.lstrip("*").strip() for line in lines]
    lines = [re.sub(number_list, "", line).strip() for line in lines]
    lines = [line for line in lines if line != ""]
    lines = [
        line
        for line in lines
        if "no specific" not in line.lower() and line.lower() != "none"
    ]

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

    if unit_mapping is not None:
        inv_map = invert_mapping(unit_mapping)
        if other:
            items = [inv_map.get(item, "other") for item in items]
        else:
            items = [inv_map.get(item, item) for item in items]
    units = sorted(list(set(items)))
    return units