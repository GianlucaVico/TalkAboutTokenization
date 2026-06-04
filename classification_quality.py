#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import random
import ipdb

TRUE_CHAR = "\x91" # PU1
FALSE_CHAR = "\x92" # PU2
SEPARATOR = "#"*40 + "\n"

CORRECT_CHAR = "v"
INCORRECT_CHAR = "x"
PARTIAL_CHAR = "/"

def create(path: str, out: str, n: int, seed: int) -> None:
    df = pd.read_json(path, lines=True)
    df = df[df["year"] >= 2016] # Old papers are excluded
    selected = df[df["topic"] == True].sample(n=n, random_state=seed)
    discarded = df[df["topic"] != True].sample(n=n, random_state=seed)
    tmp = []
    for s in [selected, discarded]:
        for row in s.iloc:
            abstract = row["abstract"]
            if abstract is None or pd.isna(abstract):
                abstract = ""
            else:
                abstract = abstract.strip()
            tmp.append((
                row["title"].strip(),
                abstract,
                row["bibtex_id"].strip(),
                TRUE_CHAR if row["topic"] == True else FALSE_CHAR,
                ))
    random.Random(seed).shuffle(tmp)
    with open(out, "xt") as f:
        for n, (title, abstract, bibtex_id, label) in enumerate(tmp):
            f.write(SEPARATOR)
            f.write(f"Sample: {n}\n")
            f.write(f"Title: {title}\n\n")
            f.write(f"Abstract:\n{abstract}\n\n")
            f.write(f"bibtex: {bibtex_id} §{label}§\n")
            f.write(f"Label: \n\n")

def parse_annotations(path: str) -> dict[str, tuple[bool, bool]]:
    """
    Returns:
        Dictionary {bibtex_id: (llm label, human_label)}        
    """
    with open(path, "rt") as f:
        content = f.read()
    content = content.replace("\ufeff", "") # Remove BOM if present
    blocks = content.split(SEPARATOR)
    blocks = [block.strip() for block in blocks if block.strip() != ""]
    annotations = {}
    for n, block in enumerate(blocks):
        lines = block.splitlines()
        bib = [line for line in lines if line.startswith("bibtex:")]
        if len(bib) != 1:
            raise ValueError(f"Found {len(bib)} bibtex lines in block {n}")
        label = [line for line in lines if line.startswith("Label:")]
        if len(label) != 1:
            raise ValueError(f"Found {len(label)} label lines in block {n}")
        label = label[0].removeprefix("Label:").strip().lower()
        if label in ["t", "true"]:
            human_label = True
        elif label in ["f", "false"]:
            human_label = False
        else:
            raise ValueError(f"Invalid label '{label}' in block {n}")
        bibtex, llm_label, *_ = bib[0].removeprefix("bibtex:").strip().split("§")
        if llm_label == TRUE_CHAR:
            llm_label = True
        elif llm_label == FALSE_CHAR:
            llm_label = False
        else:
            raise ValueError(f"Invalid LLM label '{llm_label}' in block {n}")
        annotations[bibtex] = (llm_label, human_label)
    return annotations

def evaluate(path: str) -> None:
    annotations = parse_annotations(path)
    total = len(annotations)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for bibtex, (llm_label, human_label) in annotations.items():
        if llm_label and human_label:
            tp += 1
        elif llm_label and not human_label:
            fp += 1
        elif not llm_label and not human_label:
            tn += 1
        elif not llm_label and human_label:
            fn += 1
    print(f"Total: {total}")
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {(tp + tn) / total*100:.4f}% (correct: {tp + tn}, incorrect: {fp + fn})")
    try:
        print(f"F1 Score: {2*tp / (2*tp + fp + fn)*100:.4f}%")
    except ZeroDivisionError:
        print("F1 Score: 0.0000% (UNDEFINED)")
    try:
        print(f"Precision: {tp / (tp + fp)*100:.4f}%")
    except ZeroDivisionError:
        print("Precision: 0.0000% (UNDEFINED)")
    try:
        print(f"Precision (Negative Class): {tn / (tn + fn)*100:.4f}%")
    except ZeroDivisionError:
        print("Precision (Negative Class): 0.0000% (UNDEFINED)")

    try:
        print(f"Recall (True Positive Rate): {tp / (tp + fn)*100:.4f}%")
    except ZeroDivisionError:
        print("Recall (True Positive Rate): 0.0000% (UNDEFINED)")
    try:
        print(f"False Positive Rate: {fp / (fp + tn)*100:.4f}%")
    except ZeroDivisionError:
        print("False Positive Rate: 0.0000% (UNDEFINED)")
    try:
        print(f"True Negative Rate: {tn / (fp + tn)*100:.4f}%")
    except ZeroDivisionError:
        print("True Negative Rate: 0.0000% (UNDEFINED)")
    try:
        print(f"False Negative Rate: {fn / (tp + fn)*100:.4f}%")
    except ZeroDivisionError:
        print("False Negative Rate: 0.0000% (UNDEFINED)")
    
    print(f"Confusion Matrix:\n")
    print(f"\tTP\t\tFP\t\t\t{tp}\t\t{fp}")
    print(f"\tFN\t\tTN\t\t\t{fn}\t\t{tn}")

def fix(input_path: str, output_path: str, original_path: str) -> None:
    raise NotImplementedError()

def create_info(input_path: str, output_path: str, raw_dir: str, n: int, seed: int) -> None:
    import paper_dataset
    df = pd.read_json(input_path, lines=True)
    selected = random.Random(seed).sample(range(len(df)), n)
    selected_df = df.iloc[selected]
    if os.path.exists(output_path):
        raise FileExistsError(f"Output file '{output_path}' already exists")
    
    files = {
        "units": os.path.join(raw_dir, "unit_answer.json"),
        "motivation": os.path.join(raw_dir, "motivation_answer.json"),
        "languages": os.path.join(raw_dir, "language_answer.json"),
        "lang_specific": os.path.join(raw_dir, "language_specific_answer.json"),
        "intrinsic": os.path.join(raw_dir, "evaluation_intrinsic_answer.json"),
        "extrinsic": os.path.join(raw_dir, "evaluation_extrinsic_answer.json"), 
    }

    pdfs = selected_df["pdf_url"].tolist()
    pdfs = [paper_dataset.get_pdf(url, root="/home/vico/personal_work_ms/TokSurvey/data/pdfs/") for url in pdfs]
    pdfs = [f'=HYPERLINK("{pdf}", "pdf")' for pdf in pdfs]
    selected_df["pdf_path"] = pdfs
    selected_df = selected_df[["bibtex_id", "title", "pdf_path"]]

    # Sheets: units, motivation, languages, lang. specific, evaluation-intrinsic, evaluation-extrinsic
    # Columns: index, bibtex_id, title, pdf link, raw answer, human annotation
    # Link '=hyperlink(path_to_file,"pdf")'
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, file_path in files.items():
            tmp_df = selected_df.copy()
            with open(file_path, "rt") as f:
                raw_answers = json.load(f)
                raw_answers = [raw_answers[i] for i in selected]
            tmp_df["raw_answer"] = raw_answers
            tmp_df["human_annotation"] = ""
            tmp_df.to_excel(writer, sheet_name=sheet_name)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification quality. The annotator should write T if the papers is a tokenizer paper, F if it is not.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create", help="Create the sample file")
    create_parser.add_argument("output", help="Output file for the sample")
    create_parser.add_argument("--num-samples", "-n", type=int, default=100, help="Number of samples from selected and discarded sets")
    create_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    create_parser.add_argument("--input", "-i", default="data/papers.jsonl", help="JSONL file with the papers and the 'topic' field (1/True for selected, 0/False/None for discarded)")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the classification quality")
    evaluate_parser.add_argument("annotations", help="Text file with the human annotations")

    fix_parser = subparsers.add_parser("fix", help="Fix the dataset based on the annotations")
    fix_parser.add_argument("input", help="Input file with the annotations")
    fix_parser.add_argument("output", help="Output file for the fixed dataset")  
    fix_parser.add_argument("original", help="Original dataset file (JSONL format)")

    info_extraction_parser = subparsers.add_parser("create-info", help="Create the sample file for evaluating the information extracted")
    info_extraction_parser.add_argument("output", help="Output file for the sample")
    info_extraction_parser.add_argument("--num-samples", "-n", type=int, default=10)
    info_extraction_parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
    info_extraction_parser.add_argument("--input", "-i", default="data/tokenization.jsonl")
    info_extraction_parser.add_argument("--raw-answers", default="data/raw_answers/", help="Directory with the raw answers")

    # Structure:
    # ######################################## (40)
    # Title: ...
    # \n
    # Abstract: ...
    # \n
    # bibtex: ... [PU1/PU2]
    # topic: [Annotator]
    # ######################################## (40)
    args = parser.parse_args()
    if args.command == "create":
        create(args.input, args.output, args.num_samples, args.seed)
    elif args.command == "evaluate":
        evaluate(args.annotations)
    elif args.command == "fix":
        fix(args.input, args.output, args.original)
    elif args.command == "create-info":
        create_info(args.input, args.output, args.raw_answers, args.num_samples, args.seed)