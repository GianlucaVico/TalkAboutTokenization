"""
Manually annoted mappings. To be used as `mapping.get(item, item)` in extraction functions. The kyes are a fixed list of categories; everything else is "Other".
"""

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
    ],
    "phoneme": ["phoneme", "phonemes"],
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
})