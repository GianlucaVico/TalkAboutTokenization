import os
import re
import sys
import pandas as pd
import bibfilter
import bibtexparser
import requests
import glob
import json
import textwrap
import pymupdf4llm
# Combine the raw data files into a single csv file
# Fields:
# - bibtex id 
# - title
# - author
# - year
# - abstract
# - url
# - pdf_url
# - volume (booktitle)
# - type (acl, neurips, icml, iclr, colm)
# - tldr (icml, iclr, colm)
# - keywords (icml, iclr, colm)
# - research_area (icml, iclr, colm)
# - venue (given for icml, iclr, colm, re for acl, neurips/nips for neurips)

def clean_text(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.strip()
    if len(text) == 0:
        return None
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with a single space
    return text

acl_volume = re.compile(r"\(([a-zA-Z]+) [0-9]*\)")

def _combine_acl(bibfile: str = "data/acl/anthology+abstracts.bib") -> list[dict]:
    library = bibfilter.parse(bibfile, quiet=True)
    papers = []

    default_none = bibtexparser.model.Field("missing", None)

    for block in library.blocks:
        if block.get("author") is not None and block.get("title") is not None: # It is a paper
            year = block.get("year", default_none).value
            year = int(year) if year is not None else None

            url = block.get("url", default_none).value
            pdf_url = None
            if url is not None:
                pdf_url = url[:-1] if url.endswith("/") else url
                pdf_url += ".pdf"
            
            volume = block.get("booktitle", default_none).value
            if volume is None:
                volume = block.get("journal", default_none).value
            if volume is not None:
                venue = acl_volume.search(volume)
                if venue is not None:
                    venue = venue.group(1)

            paper = {
                "bibtex_id": block.key,
                "title": block.get("title").value,
                "author": clean_text(block.get("author").value),
                "year": year,
                "abstract": block.get("abstract", default_none).value,
                "type": "acl",
                "url": url,
                "pdf_url": pdf_url,
                "volume": volume,
                "tldr": None,
                "keywords": None,
                "research_area": None,
                "venue": venue
            }
            papers.append(paper)
    return papers

def _combine_neurips(bibfile: str = "data/neurips/neurips.bib") -> list[dict]:
    library = bibfilter.parse(bibfile, quiet=True)
    papers = []

    default_none = bibtexparser.model.Field("missing", None)

    for block in library.blocks:
        if block.get("author") is not None and block.get("title") is not None: # It is a paper
            year = block.get("year", default_none).value
            year = int(year) if year is not None else None

            pdf_url = block.get("url", default_none).value
            url = None # Not scraped

            paper = {
                "bibtex_id": block.key,
                "title": block.get("title").value,
                "author": clean_text(block.get("author").value),
                "year": year,
                "abstract": block.get("abstract", default_none).value,
                "type": "neurips",
                "url": url,
                "pdf_url": pdf_url,
                "volume": block.get("booktitle", default_none).value,
                "tldr": None,
                "keywords": None,
                "research_area": None,
                "venue": "neurips"
            }
            papers.append(paper)
    return papers

def _combine_openreview(folder: str, type_: str) -> list[dict]:
    papers = []
    default_none = bibtexparser.model.Field("missing", None)
    for file in glob.glob(os.path.join(folder, "*.json")):
        with open(file, "r") as f:
            data = json.load(f)
        
        if "title" in data and "authors" in data:
            year = data.get("year", None)
            if year is not None:
                year = int(year)
            keywords = data.get("keywords", None)
            if isinstance(keywords, list):
                keywords = ", ".join(keywords)
            if len(keywords) == 0:
                keywords = None
            tldr = data.get("TLDR", None)
            if len(tldr) == 0:
                tldr = None
            
            bibtex = data.get("bibtex", None)
            if bibtex is not None and len(bibtex) != 0:
                try:
                    bibtex = re.sub(r"\{\\\w\{(\w)\}\}", r"\1", bibtex)  # {\c{C}}
                    bibtex = re.sub(r"\{(\\\W)?\\?(\w+)\}", r"\2", bibtex) # {\"o} {\'\i} {\i}
                    bibtex = bibtex.replace(r"{\&}", "&")  
                    lib = bibtexparser.parse_string(bibtex)
                    block = lib.blocks[0]
                    bibid = block.key
                    url = block.get("url", default_none).value
                    volume = block.get("booktitle", default_none).value
                except Exception as e:
                    print(f"Error parsing bibtex for {data.get('id')}: {bibtex}")
                    raise e
            else:
                bibid = None
                url = None
                volume = None

            paper = {
                "bibtex_id": f"{data.get('id')}_{bibid}",
                "title": data.get("title", None),
                "author": clean_text(" and ".join(data.get("authors", []))),
                "year": year,
                "abstract": data.get("abstract", None),
                "type": type_,
                "url": url,
                "pdf_url": data.get("pdf", None),
                "volume": volume,
                "tldr": tldr,
                "keywords": keywords,
                "research_area": ", ".join(data.get("research_area", [])),
                "venue": data.get("venue", None)
            }
            papers.append(paper)
    return papers

def create(file: str = "data/papers.jsonl") -> None:
    acl = _combine_acl()
    neurips = _combine_neurips()
    icml = _combine_openreview("data/icml/papers", "icml")
    iclr = _combine_openreview("data/iclr/papers", "iclr") 
    colm = _combine_openreview("data/colm/papers", "colm")
    papers = acl + neurips + icml + iclr + colm
    papers = pd.DataFrame(papers).drop_duplicates(subset=["bibtex_id"])
    papers.to_json(file, orient="records", lines=True)


def load(file: str = "data/papers.jsonl") -> pd.DataFrame:
    df = pd.read_json(file, lines=True, orient="records")
    return df

def pdf_url_to_file(url: str) -> str:
    return url.split("https://")[-1].replace("/", "")

def get_pdf(url: str, root: str = 'data/pdfs/') -> str:    
    path = pdf_url_to_file(url)
    path = os.path.join(root, path)
    if not os.path.exists(path):
        # print(f"Downloading {url} to {path}...", file=sys.stderr)
        os.makedirs(root, exist_ok=True)
        pdf = requests.get(url)    
        if pdf.status_code == 200:
            with open(path, "wb") as f:
                f.write(pdf.content)
        else:
            raise ValueError(f"Failed to download PDF from {url}, status code: {pdf.status_code}")
    return path

def to_markdown(pdf_path: str, truncate_references: bool = True, remove_urls: bool = False, normalize_white: bool = True) -> str:
    pages = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, ignore_graphics=True, ignore_images=True, use_ocr=False)
    md = pages[0]['text'] if len(pages) > 0 else ""
    md = re.sub(r'\n+[0-9]+\n+', ' ', md) # Remove page numbers
    for page in pages[1:]:
        tmp = page['text']
        tmp = re.sub(r'\n+[0-9]+\n+', ' ', tmp)
        tmp = re.sub(r'^\W*\*\*.*\*\*\n+', ' ', tmp) # Remove title
        md += tmp 
    if truncate_references:
        # Truncate at References or Bibliography
        i = md.find('\n**References')
        if i != -1:
            md = md[:i]
    if remove_urls:
        md = re.sub(r'\[.*\]\(.*\)', '', md)  
    if normalize_white:
        # md = re.sub(r'\n\s*\n', '\n\n', md)
        # md = re.sub(r'\n+', '\n', md)
        md = re.sub(r' +', ' ', md)
    return md.strip()

def get_markdown(url: str, root: str = 'data/pdfs/', cache: bool=True) -> str:
    pdf_path = get_pdf(url, root)
    if cache:
        md_path = pdf_path + ".md"
        if os.path.exists(md_path):
            with open(md_path, "r") as f:
                return f.read()
        else:
            md = to_markdown(pdf_path)
            with open(md_path, "w") as f:
                f.write(md)
            return md
    else:
        return to_markdown(pdf_path)    

EXCLUDE_SECTIONS = [
    'reference', 'references', 'bibliography', 'acknowled', 
    'appendix', 'supplement', 'contributions', 'ethic', 
    # 'experiment', 'result'
]

def split_sections(md: str, num: bool = False, exclude_list: list[str] | None = EXCLUDE_SECTIONS) -> list[str]:
    # ** 1 Introduction ** | **1** **Introduction** style
    r = r'(?=#* *\*\*[0-9]+.*\*\*)' if num else r'(?=#* *\*\*.*\*\*)'
    parts = re.split(r, md)
    parts = [p.strip() for p in parts if len(p.strip()) > 3] # Remove almost empty sections
    if len(parts) == 1:
        # \n1 Introduction\n | \n1 I NTRODUCTION\n style
        r = r'(?=\n+[0-9]+[A-Za-z &]+\n+)' if num else r'(?=\*\*.*\*\*)'
        parts = re.split(r, md)
        parts = [p.strip() for p in parts if len(p.strip()) > 0]

    if exclude_list is not None:        
        titles = [
            re.search(r'[a-z &]+', p.split('\n')[0].lower())
            for p in parts
        ]            
        titles = [i.group().strip() for i in titles if i is not None]
        titles = [any(ex in t for ex in exclude_list) for t in titles]
        parts = [p for p, t in zip(parts, titles) if not t]
    return parts 
    

def print_sample(row):
    print("=" * 80)
    print(f"Title: {row['title']}")
    print(f"Author(s): {row['author']}")
    print(f"Year: {row['year']}")
    print(f"Venue: {row['venue']} ({row['type'].upper()})")
    print(f"Volume: {row['volume']}")
    print("-" * 80)
    print(f"Abstract:")
    abstract = row['abstract'] or "No abstract available"
    wrapped_abstract = textwrap.fill(abstract, width=75, initial_indent="   ", subsequent_indent="   ")
    print(wrapped_abstract)
    print("=" * 80)
    print()

def sample(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.sample(n=n, random_state=42)

def filter_topic_re(df: pd.DataFrame) -> pd.Series: # OUTDATED/UNUSED
    keywords = re.compile(r"token|sub-?word|segment")
    return df["abstract"].apply(lambda x: x is not None and re.search(keywords, x.lower()) is not None)

def not_duplicate_mask(df: pd.DataFrame) -> pd.Series:
    return ~df.duplicated(subset=["title"], keep='first')

if __name__ == "__main__":
    if not os.path.exists("data/papers.jsonl"):
        create()
        df = load()
        print(df.head())
        print(f"Total papers: {len(df)}")
    else:
        print("Dataset already exists. Looking for new papers...") 
        create(file="data/papers_new.jsonl")
        df = load("data/papers_new.jsonl")
        old_df = load("data/papers.jsonl")
        new_df = df[~df["bibtex_id"].isin(old_df["bibtex_id"])]
        print(new_df.head())
        print(f"New papers found: {len(new_df)}")
        if len(new_df) > 0:
            new_df.to_json("data/papers_diff.jsonl", orient="records", lines=True)
