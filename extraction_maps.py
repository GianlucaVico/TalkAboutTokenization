"""
Manually annoted mappings. To be used as `mapping.get(item, item)` in extraction functions. The kyes are a fixed list of categories; everything else is "Other".
"""
import re

class _hashed_dict(dict):
    def __init__(self, hash_, dict_):
        super().__init__(dict_)
        self._hash = hash_
    def __hash__(self):
        return self._hash

UNIT_MAPPING = _hashed_dict(hash("unit"), {
    "morpheme": [
        "affix",
        "inflectional group",
        "morpheme",
        "morphemes",
        "morpheme-like units",
        "root token",
        "suffix token",
        "template stem token",
        "prefix token",
        "morpheme‑like units"
    ],
    "syllable": [
        "orthographic syllable",
        "syllable",
        "syllable units",
        "syllables",
    ],
    "bytes": [
        "byte",
        "bytes",
        "byte‑level units",
        "utf‑8 byte",
    ],
    "character": [
        "character",
        "characters",
        "chinese character",
        "japanese character",
        "grapheme",
        "individual character",
        "individual characters",
        "unicode character",
        "unicode code point",
        "character n‑grams",
        "character sequence",
        "character trigrams",
    ],
    "sub-character": [
        "jamo",
        "sub-characters",
        "subcharacter",
        "sub‑character",
        "sub‑characters",
    ],
    "word": [
        "gloss",
        "lemma",
        "whitespace delimiter",
        "word",
        "word token",
        "words",
        "word‑form",
        "space character",
        "词",
    ],
    "none": ["none"],
    "subword": [
        "subword",
        "subword token",
        "subword tokens",
        "subword unit",
        "subword units",
        "subwords",
        "word‑piece",
        "wordpiece",
        "word piece",
        "token",
        "token chunk",
        "token pairs",
        "tokens",
        "text chunks",
        "sub‑segment"
    ],
    "phoneme": ["phoneme", "phonemes", "acoustic tokens", "short‑time audio frames"],
    "biological": [
        "amino‑acid substitution token",
        "base",
        "nucleotide",
        "protein family token",
        "single nucleotide",
        "standard amino‑acid token",
        "rna family token",
        "intonation units",
    ],
    "sentence": [
        "sentence",
        "phrase",
        "phrase tokens",
    ],
    "image": ["image patch"],
    "patch": [
        "patches",
    ],
    "pixels": ["pixels"],
})

def task_mapping(task: str) -> str:
    task = task.lower()
    # Machine translation
    mt_keywords = ["machine translation", "translation", "wmt", "flores", "→"]
    if any(keyword in task for keyword in mt_keywords):
        return "machine translation"
    if re.search(r"\bmt\b", task):
        return "machine translation"    
    
    # Summarization
    summarization_keywords = ["summar", "xsum", "crossum", "sum", "cnn"]
    if any(keyword in task for keyword in summarization_keywords):
        return "summarization"
    

    # Part of speech tagging
    if "part-of-speech" in task or "part of speech" in task or "pos tagging" in task:
        return "part of speech tagging"
    if re.search(r"\bpos\b", task):
        return "part of speech tagging"
    
    # Named entity recognition
    ner_keywords = ["named entity recognition", "named entity extraction", "conll", "masakhaner"]
    if any(keyword in task for keyword in ner_keywords):
        return "named entity recognition"
    if re.search(r"\bner\b", task):
        return "named entity recognition"
    
    # Language modeling
    lm_keywords = ["language modeling", "language modelling", "text continuation", "next token prediction", "lambada", "babyslm", "generation", "hellaswag", "word prediction"]
    if any(keyword in task for keyword in lm_keywords):
        return "language modeling"
    if re.search(r"\blm\b", task):
        return "language modeling"
    
    # Question answering (including vision qa)
    qa_keywords = ["question answering", "mmlu", "piqa", "sib-200", "chartqa", "textvqa", "okvqa", "squad", "textbookqa", "searchqa", "newsqa", "x-csqa", "pubmedqa", "massively multilingual nlu", "boolq", "arc", "winogra", "jcqa", "obqa", "tydiqa", "qqp", "triviaqa", "openbookqa", "truthfulqa", "commonsenseqa", "afqmc", "wikidataqa", "commonsenseqa", "belebele", "quad", "natural questions"]
    if any(keyword in task for keyword in qa_keywords):
        return "question answering"
    if re.search(r"\bqa\b", task):
        return "question answering"
    if task.endswith("qa"):
        return "question answering"
    
    if "morph" in task:
        return "morphological task"

    # Other common tasks
    if "dependency" in task or "constituency" in task or task=="ud":
        return "dependency parsing"
    if "coref" in task or "entity linking" in task:
        return "coreference resolution"
    
    sentiment_keywords = ["sentiment", "polarity", "hate", "irony", "offensive", "stance", "sst"]
    if any(keyword in task for keyword in sentiment_keywords):
        return "sentiment analysis"
    if "identification" in task:
        return "language identification"
    
    # Natural language inference
    nli_keywords = ["natural language inference", "nli", "rte", "mrpc"]
    if any(keyword in task for keyword in nli_keywords):
        return "natural language inference"    

    # Some generic tasks
    if "classification" in task or "accuracy" in task or "f1" in task or "precision" in task or "recall" in task:
        return "classification"
    if "clustering" in task:
        return "clustering"
    if "information retrieval" in task or "retrieval" in task:
        return "information retrieval"
    if "regression" in task:
        return "regression"
    
    # Generic bias evaluation
    bias_keywords = ["bias", "fairness", "stereotype", "discrimination", "gender", "racial"]
    if any(keyword in task for keyword in bias_keywords):
        return "bias evaluation"

    # Segmentation/tokenization
    if "segmentation" in task or "tokenization" in task or "chunking" in task:
        return "segmentation"
    
    # Vision tasks
    if "image" in task or "vision" in task or "video" in task or "visual" in task:
        return "vision"
    # Speech tasks
    if "speech" in task or "audio" in task or "asr" in task or "spoken" in task or "phon" in task:
        return "speech"
    
    if "correction" in task or "restoration" in task:
        return "error correction"

    if "math" in task or "arithmetic" in task:
        return "math"
    if "time series" in task:
        return "time series analysis"
    
    if "cola" in task or "acceptability" in task or "copa" in task or "readability assessment" in task or "languistic complexity" in task:
        return "linguistic acceptability"
    if "mbpp" in task or "code generation" in task or "humaneval" in task or "blimp" in task:
        return "code generation"
    if "similarity" in task:
        return "similarity"
    return None
    