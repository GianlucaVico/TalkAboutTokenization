import re
import itertools
import pycountry

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
        elif item in ["patch tokens", "patches"]:
            tmp.append("patches")
        elif item in ["image tokens", "visual tokens"]:
            tmp.append("image tokens")
        elif item in ["word", "words"]:
            tmp.append("words")
        elif "\n" in item or ")" in item:
            pass
        else:
            tmp.append(item)

    units = sorted(list(set(tmp)))

    return units