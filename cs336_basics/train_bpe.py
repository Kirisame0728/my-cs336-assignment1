from pathlib import Path
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_docs(input_path: str, special_token: str = "<|endoftext|>") -> list[str]:
    text = Path(input_path).read_text(encoding="utf8")
    docs = [doc.strip() for doc in text.split(special_token) if doc.strip()]
    return docs

def pre_tokenize(doc: str) -> list[str]:
    tokens = [m.group() for m in re.finditer(PAT, doc)]
    return tokens

def pretoken_to_token_seq(pretoken: str) -> tuple[bytes, ...]:
    utf8_encode = pretoken.encode("utf-8")
    return tuple(bytes([b]) for b in utf8_encode)

def build_token_seq_freqs(docs: list[str]) -> dict[tuple[bytes, ...], int]:
    token_seq_freqs = Counter()
    for doc in docs:
        for tokens in pre_tokenize(doc):
            token_seq = pretoken_to_token_seq(tokens)
            token_seq_freqs[token_seq] += 1

    return dict(token_seq_freqs)

