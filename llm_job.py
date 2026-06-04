#!/usr/bin/env python3
import paper_dataset
import llm_filter
import json
import argparse
import os
import pandas as pd
from transformers import utils
utils.logging.set_verbosity_error()
import tqdm
import ipdb

BACKEND = "api"
YEAR_MIN = 2014
YEAR_MAX = 2025
THRESHOLD_PDF_SIZE = 5_000 # bytes
ZERO_SHOT = True

def filter_ds(original_ds: str, tokenization_ds: str, tokfree: bool = False, append: bool = False) -> None:
    if not os.path.exists(original_ds):
            raise FileNotFoundError("Paper dataset not found")
    if os.path.exists(tokenization_ds) and not append:
        raise FileExistsError("Tokenization dataset already exists.")            

    df = paper_dataset.load(original_ds)
    keywords = llm_filter.RE_FILTER

    # Select years
    mask = (df['year'] >= YEAR_MIN) & (df['year'] <= YEAR_MAX)
    
    # Remove papers with no abstract or title
    check_str = lambda x: isinstance(x, str) and x.strip() != ""
    mask = mask & df.apply(lambda x: check_str(x['abstract']) and check_str(x['title']), axis=1)

    # Select papers matching keywords in title or abstract
    check_str = lambda x: isinstance(x, str) and keywords.search(x.lower()) is not None
    mask = mask & df.apply(lambda x: check_str(x['title']) or check_str(x['abstract']), axis=1)
    mask = mask | df.apply(lambda x: x['venue'] == "TokShop", axis=1) # TokShop was in 2025 despite bibtex says 2026 and it is tokenization
    tokenization = df[mask]
    tokenization = tokenization.reset_index(drop=True)
    df["re_select"] = mask

    abstracts = tokenization['abstract'].tolist()
    titles = tokenization['title'].tolist()
    os.makedirs("data/raw_answers/", exist_ok=True)
    if tokfree:
        filter_mask = list(llm_filter.filter_tokfree(abstracts, titles, backend=BACKEND, zero_shot=ZERO_SHOT, tqdm_disable=False, file="data/raw_answers/filter_tokfree_answer.txt"))
        file_name = "filter_tokfree_answer.json"
    else:
        filter_mask = list(llm_filter.filter_topic(abstracts, titles, backend=BACKEND, zero_shot=ZERO_SHOT, tqdm_disable=False, file="data/raw_answers/filter_answer.txt"))
        file_name = "filter_answer.json"
    with open(f"data/raw_answers/{file_name}", "wt") as f:
        json.dump(filter_mask, f, indent=2)
    
    tokenization['topic'] = filter_mask 
    # Check if the pdf exists
    def discard(url: str, topic: bool, threshold=THRESHOLD_PDF_SIZE, root='data/pdfs/'):        
        if not topic: # Not tokenization
            return True
        if url == '':
            return True
        try:
            paper_dataset.get_markdown(url)
        except (ValueError, paper_dataset.pymupdf4llm.pymupdf.FileDataError) as e:
            return True
        md = os.path.join(root, paper_dataset.pdf_url_to_file(url) + '.md')
        return (not os.path.exists(md)) or (os.path.getsize(md) < threshold)

    discard_mask = []
    for row in tqdm.tqdm(tokenization.iloc, total=len(tokenization), desc="Checking PDFs"):
        url = row['pdf_url']
        topic = row['topic']
        discard_mask.append(discard(url, topic))

    filter_mask = filter_mask & (~pd.Series(discard_mask))
    tokenization['topic'] = filter_mask
    
    # Merge the tokenization DataFrame with the original DataFrame adding the column 'topic' (bool)
    df = df.merge(tokenization[['bibtex_id', 'topic']], on='bibtex_id', how='left')
    
    # Save df to jsonl file
    if not tokfree:
        df.to_json(original_ds, orient='records', lines=True)
    
    tokenization = tokenization[tokenization['topic']]
    
    if append:        
        existing_tokenization = paper_dataset.load(tokenization_ds)
        tokenization = pd.concat([existing_tokenization, tokenization], ignore_index=True)
        # TODO fix re_select and topic in df
        df[df['re_select']][filter_mask] = True
        df.to_json(original_ds, orient='records', lines=True)
        
    tokenization.to_json(tokenization_ds, orient='records', lines=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-ds", type=str, default="data/papers.jsonl", help="Path to the list of papers")
    parser.add_argument("--tokenization-ds", type=str, default="data/tokenization.jsonl", help="Path to the tokenization papers")
    parser.add_argument("--filter", action="store_true")    
    parser.add_argument("--units", action="store_true")
    parser.add_argument("--motivation", action="store_true")
    parser.add_argument("--language", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--tokfree", action="store_true")
    args = parser.parse_args()
    
    if args.filter:
        filter_ds(args.original_ds, args.tokenization_ds, tokfree=False)
    if args.tokfree:
        filter_ds(args.original_ds, args.tokenization_ds, tokfree=args.tokfree, append=args.filter)


    print("Loading tokenization dataset...")
    tokenization = paper_dataset.load(args.tokenization_ds)
    print(f"Tokenization dataset loaded with {len(tokenization)} papers.")
    os.makedirs("data/raw_answers/", exist_ok=True)
    # UNITS
    if args.units:
        print("=== UNIT EXTRACTION ===")
        all_answers_unit = list(
            llm_filter.get_units(
                tokenization, backend=BACKEND, file="data/raw_answers/unit_answer.txt"
            )
        )
        with open("data/raw_answers/unit_answer.json", "wt") as f:
            json.dump(all_answers_unit, f, indent=2)


    # MOTIVATIONS
    if args.motivation:
        print("=== MOTIVATION EXTRACTION ===")
        all_answers_motivation = list(
            llm_filter.get_motivation(
                tokenization,
                backend=BACKEND,
                file="data/raw_answers/motivation_answer.txt",
            )
        )
        with open("data/raw_answers/motivation_answer.json", "wt") as f:
            json.dump(all_answers_motivation, f, indent=2)

    # LANGUAGES
    if args.language:
        print("=== LANGUAGE EXTRACTION ===")
        all_answers_language = list(
            llm_filter.get_languages(
            tokenization,
            backend=BACKEND,
            file="data/raw_answers/language_answer.txt",
            )
        )
        with open("data/raw_answers/language_answer.json", "wt") as f:
            json.dump(all_answers_language, f, indent=2)

        print("=== LANGUAGE SPECIFICITY EXTRACTION ===")
        all_answers_language_specific = list(
            llm_filter.get_languages_specific(
            tokenization,
            backend=BACKEND,
            file="data/raw_answers/language_specific_answer.txt",
            )
        )
        with open("data/raw_answers/language_specific_answer.json", "wt") as f:
            json.dump(all_answers_language_specific, f, indent=2)

    # EVALUATION    
    if args.evaluation:
        print("=== INTRINSIC EVALUATION EXTRACTION ===")
        all_answers_evaluation_intrinsic = list(
            llm_filter.get_evaluation_intrinsic(
            tokenization,
            backend=BACKEND,
            file="data/raw_answers/evaluation_intrinsic_answer.txt",
            )
        )
        with open("data/raw_answers/evaluation_intrinsic_answer.json", "wt") as f:
            json.dump(all_answers_evaluation_intrinsic, f, indent=2)
        
        print("=== EXTRINSIC EVALUATION EXTRACTION ===")
        all_answers_evaluation_extrinsic = list(
            llm_filter.get_evaluation_extrinsic(
            tokenization,
            backend=BACKEND,
            file="data/raw_answers/evaluation_extrinsic_answer.txt",
            )
        )
        with open("data/raw_answers/evaluation_extrinsic_answer.json", "wt") as f:
            json.dump(all_answers_evaluation_extrinsic, f, indent=2)
