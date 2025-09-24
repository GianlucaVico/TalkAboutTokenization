from collections.abc import Callable
from typing import Any
import bibtexparser
import bibtexparser.model
import ipdb
import tqdm
import re

keywords = re.compile(r"token|sub-?word|segment")

def bibfilter(library, filters: dict[str: Callable[[Any], bool]], *, quiet: bool = True) -> list:
    selected = []
    for block in tqdm.tqdm(library.blocks, disable=quiet):
        for key, filter in filters.items():
            field = block.get(key)
            if field is None or not filter(field.value):
                break
        else: # checked all filters
            selected.append(block)
    return selected

def parse(file_path: str, *, quiet: bool = True) -> bibtexparser.Library:
    with open(file_path) as f:
        raw = f.read().strip()
    parts = re.split(r"}\n+@", raw)
    parts[0] = parts[0].strip()[1:] # remove the @ at the beginning
    parts[-1] = parts[-1].strip()[:-1] # remove the } at the end
    parts = [i.replace("@", "[at]") for i in parts]
    parts = ["@"+i+"}" for i in parts]

    l = bibtexparser.Library()
    check_fn = lambda x: isinstance(x, bibtexparser.model.Entry)
    for i, part in enumerate(tqdm.tqdm(parts, disable=quiet)):
        tmp = bibtexparser.parse_string(part)
        if all(map(check_fn, tmp.blocks)):
            l.add(tmp.blocks)
        else:
            # raise ValueError(f"Invalid entry ({i}):\n {part}")        
            print(f"Invalid entry ({i}):\n {part}")
            continue
    return l



if __name__ == "__main__":
    filters = {
        "year": lambda x: int(x) >= 2016,
        "abstract": lambda x: re.search(keywords, x.lower()) is not None
    }
    
    # ACL
    library = parse("data/acl/anthology+abstracts.bib", quiet=False)
    acl_selected = filter(library, filters, quiet=False)
    print(f"ACL: {len(library.blocks)} -> {len(acl_selected)}")

    # NeurIPS
    library = parse("data/neurips/neurips_20250610.bib", quiet=False)
    neurips_selected = filter(library, filters, quiet=False)
    print(f"NeurIPS: {len(library.blocks)} -> {len(neurips_selected)}")

