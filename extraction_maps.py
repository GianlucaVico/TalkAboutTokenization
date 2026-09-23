"""
Manually annoted mappings. To be used as `mapping.get(item, item)` in extraction functions. The kyes are a fixed list of categories; everything else is "Other".
"""

import re
import functools


class _hashed_dict(dict):
    def __init__(self, hash_, dict_):
        super().__init__(dict_)
        self._hash = hash_

    def __hash__(self):
        return self._hash


@functools.cache
def invert_mapping(mapping: dict[str, list[str]]) -> dict[str, str]:
    inverted = {}
    for key, values in mapping.items():
        for value in values:
            inverted[value] = key
    return inverted


# Exact match
UNIT_MAPPING = _hashed_dict(
    hash("unit"),
    {
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
            "morpheme‑like units",
            "affix (clitic)",
            "base (stem)",
            "inflectional group (ig)",
            "paradigm components (roots suffixes)",
            "subword (morpheme)",
            "sub‑segment (morpheme)",
            "subwords (morphemes)",
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
            "byte (utf‑8)",
            "byte‑level units (fallback)",
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
            "character (as the smallest element that can be split)",
            "character (grapheme)",
            "character (hangul syllable)",
            "character (including inline flags)",
            "character (symbol)",
            "character (unicode code point)",
            "character n‑grams (subword sequences)",
            "character trigrams (character n‑grams)",
            "characters (bytes)",
            "characters (graphemes)",
            "chinese character (han)",
            "grapheme (grapheme cluster)",
            "individual characters (fallback)",
            "symbol",
            "symbol (i.e., a character from the source alphabet)",
            "symbols",
            "unicode character (code point)",
        ],
        "sub-character": [
            "jamo",
            "sub-characters",
            "subcharacter",
            "sub‑character",
            "sub‑characters",
            "cv (jamo: consonant‑vowel)",
            "jamo (individual hangul sub‑character components)",
            "sub‑character (ideograph)",
            "sub‑character (stroke)",
            "subwords (jamo)",
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
            "bigram (two‑word collocation)",
            "lemma (base form of a word)",
            "words (space‑delimited tokens)",
            "word‑form (candidate segment)",
            "trigram (three‑word collocation)",
            "token (word)",
            "unigram (single word)",
            "word (eojeol)",
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
            "subword (bpe token)",
            "subword (bpe) token",
            "subword (byte‑pair) token",
            "subword (byte‑pair‑encoded) token",
            "subword (sentence‑piece)",
            "subword (subword token)",
            "subword (token)",
            "subword (word piece)",
            "subword (wordpiece token)",
            "subword (wordpiece)",
            "subword (wordpiece) token",
            "subword (wordpiece) tokens",
            "subword (word‑piece)",
            "subword (word‑piece) token",
            "subword token (bpe)",
            "subword token (bpe‑derived)",
            "subword token (byte‑pair encoding)",
            "subword token (e.g., bpe/wordpiece unit)",
            "subword (character‑level) units",
            "subword (character\u202fn‑gram)",
            "subword (e.g., bpe/wordpiece)",
            "subword token (word‑piece)",
            "subword tokens (bpe)",
            "subword tokens (subtokens)",
            "subwords (bpe)",
            "subwords (byte‑pair encoding)",
            "subwords (subtokens)",
            "subwords (token‑piece tokens)",
            "token (subword token)",
            "tokens (subwords)",
            "wordpiece (subword)",
        ],
        "token (generic)": [
            "token",
            "token chunk",
            "token pairs",
            "tokens",
            "special token (e.g., <cite>)",
        ],
        "phonetics": [
            "phoneme",
            "phonemes",
            "acoustic tokens",
            "short‑time audio frames",
            "acoustic tokens (vector‑quantized codes)",
            "intonation units (prosodic speech segments)",
            "short‑time audio frames (e.g., 10\u202fms windows)",
        ],
        "biological": [
            "amino‑acid substitution token",
            "nucleotide",
            "protein family token",
            "single nucleotide",
            "standard amino‑acid token",
            "rna family token",
            "intonation units",
            "amino‑acid substitution token (e.g., a→v)",
            "k‑mer",
            "k‑mer (contiguous subsequence of nucleotides, variable length 1–10)",
            "nucleotide (single base)",
            "protein family (pfam) token",
            "rna family (rfam) token",
            "standard amino‑acid token (canonical residue)",
            "subword token (variable‑length nucleotide fragment)",
        ],
        "sentence": [
            "sentence",
            "phrase",
            "phrase tokens",
            "phrase (multi‑word token)",
            "phrase (n‑gram) tokens",
            "text chunks (e.g., sentences or fixed‑length spans of tokens)",
        ],
        "image": ["image patch"],
        "patch": [
            "patches",
            "other (continuous values)",
        ],
        "pixels": ["pixels"],
    },
)

# Exact match
METRIC_MAPPING = _hashed_dict(
    hash("metric"),
    {
        "precision (generic)": ["precision", "p"],
        "recall (generic)": ["recall", "in vocabulary recall", "r"],
        "f1 score (generic)": ["f1", "f1 score", "f score", "fscore", "f measure", "word f1", "f", "f1 measure", "f 1 score"],
        "accuracy (generic)": [
            "accuracy",
            "acc",
            "token level accuracy",
            "word accuracy",
            "sentence accuracy",
            "exponent accuracy",
            "classification accuracy",
            "form accuracy",            
            "error rate", # 1 - accuracy
            "a", # a@...
            "sequence accuracy",
        ],
        "oov": [
            "oov recall",
            "oov rate",
            "out of vocabulary recall",
            "out of vocabulary rate",
        ],
        "exact match": [
            "match",
            "sentence level match",
            "match accuracy",
        ],
        "rényi/shannon": [
            "rényi entropy",
            "renyi entropy",
            "rényi efficiency",
            "renyi efficiency",
            "shannon efficiency",
            "description length"
        ],
        "correlation": [
            "spearmans ρ",
            "spearmans rank correlation",
            "spearman correlation",
        ],
        "fertility": [
            "fertility",
            "subword fertility",
            "tokens per word",
            "subwords per word",
        ],
        "distance": [
            "edit distance",
            "levenshtein distance",
        ],
        "morphology": [
            "boundary precision",
            "boundary recall",
            "boundary f1",
            "boundary accuracy",
            "boundary precision recall",
            "morphscore",
            "morpheme precision",
            "morpheme recall",
            "morpheme f1",
            "morpheme accuracy",
            "morphological coverage",
            "emma",
            "emma 2",
        ],
        "segmentation": [
            "segmentation accuracy",
            "segmentation precision",
            "segmentation recall",
            "word segmentation f1",
            "tokenization f1",
            "segmentation f1",
            "split prediction accuracy",
            "sentence segmentation f1",
        ],
        "perplexity": ["perplexity", "ppl", "language modeling loss", "pseudo perplexity"],
        "tokenization parity": ["tokenization parity", "parity"],
        "overlap/similarity": [
            "overlap",
            "jaccard similarity",
            "vocab overlap",
            "frequency weighted overlap",
            "cosine similarity",
            "dice coefficient",
        ],
        "token length": [
            "token length distribution",
            "token length",
            "subword length",
        ],
        "token count": ["tokens", "corpus token count", "sequence length"],
        "compression": [
            "compression rate",
            "compression ratio",
            "bits per byte",
            "bytes per token",
        ],
        "coverage": [
            "coverage",
            "vocabulary coverage",
            "vocabulary usage"
        ],
        "vocabulary size": [
            "vocabulary size",
        ],
        "type-token ratio": [
            "type token ratio",
        ],
        "cognitive plausibility": [
            "cognitive plausibility",
        ],
        "bleu": [
            "bleu", "bleu score",
        ],
        "auc": [
            "auc",
        ],
        "visual task": ["lpips"],
        "rank": ["rank"],
        "continued words": ["continued words"],
    },
)

# Is in
DATASET_MAPPING = _hashed_dict(
    hash("dataset"),
    {
        "machine translation": ["wmt", "flores"],
        "natural language inference": ["xnli", "mnli", "qnli", "wnli", "rte", "nli"],
        "question answering": [
            "mmlu",
            "boolq",
            "arc",
            "openbookqa",
            "triviaqa",
            "truthfulqa",
        ],
        "sentiment analysis": ["sst"],
        "commonsense reasoning": [
            "commonsenseqa",
            "hellaswag",
            "piqa",
            "winogra",
            "copa",
        ],
        "language modeling": ["lambada"],
        "code generation": ["mbpp", "humaneval"],
        "paraphrase detection": ["paws", "qqp", "mrpc"],
        "character tasks": ["cute"],
        "math": ["gsm8k", "math"],
        "linguistic judgment": ["cola", "blimp"],
        "similarity": [
            "sts",
        ],
        "summarization": [
            "cnn",
        ],  # others already coverted
        "reading comprehension": ["squad", "xquad", "race", "drop", "quad"],
    },
)


def task_mapping(task: str) -> str:
    task = task.lower()

    # Recognizable tasks
    if "word segmentation" in task:
        return "word segmentation"

    if "translation" in task or "→" in task:
        return "machine translation"

    if "tagging" in task:
        return "part-of-speech tagging"

    if "entity recognition" in task:
        return "named entity recognition"

    if "dependency" in task:
        return "dependency parsing"

    if "sentiment" in task:
        return "sentiment analysis"

    if "modeling" in task:
        return "language modeling"

    if "inference" in task:
        return "natural language inference"

    if "sum" in task:
        return "summarization"

    if "similarity" in task:
        return "similarity (generic)"

    if "question answer" in task:
        return "question answering"

    if "text classification" in task or "topic classification" in task:
        return "text classification"

    if "automatic speech recognition" in task:
        return "automatic speech recognition"

    if "lemmatization" in task:
        return "lemmatization"

    if "common sense" in task or "commonsense" in task:
        return "commonsense reasoning"

    if "code generation" in task:
        return "code generation"

    if "paraphras" in task:
        return "paraphrase detection"

    if "character" in task:
        return "character tasks"

    if "math" in task:
        return "math"

    if "acceptability" in task:
        return "linguistic judgment"

    if "comprehension" in task:
        return "reading comprehension"

    # Common abbreviations
    if re.search(r"\bmt\b", task):
        return "machine translation"
    if re.search(r"\bpos\b", task):
        return "part-of-speech tagging"
    if re.search(r"\bner\b", task):
        return "named entity recognition"
    if re.search(r"\blm\b", task):
        return "language modeling"
    if re.search(r"\bqa\b", task):
        return "question answering"
    if re.search(r"\bnli\b", task):
        return "natural language inference"
    if re.search(r"\bsum\b", task):
        return "summarization"

    # Common dataset (>threshold) mentions
    ds_map = invert_mapping(DATASET_MAPPING)
    for ds, category in ds_map.items():
        if ds in task:
            return category
    return None
