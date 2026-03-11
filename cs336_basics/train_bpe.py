from pathlib import Path
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_docs(input_path: str, special_tokens: list[str]) -> list[str]:
    text = Path(input_path).read_text(encoding="utf8")
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    docs = [doc for doc in re.split(pattern, text) if doc != ""]
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

# Byte Pair Encoding part
def count_pairs(token_seq_freqs: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    pair_freqs = Counter()
    for token_seq, num in token_seq_freqs.items():
        if len(token_seq) < 2:
            continue
        else:
            for pair in zip(token_seq, token_seq[1:]):
                pair_freqs[pair] += num

    return dict(pair_freqs)

def get_max_pairs(pair_freqs: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes] | None:
    if not pair_freqs:
        return None
    return max(pair_freqs.items(), key=lambda x: (x[1], x[0]))[0]

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {}
    idx = 0
    for st in special_tokens:
        vocab[idx] = st.encode("utf-8")
        idx += 1
    for i in range(256):
        vocab[idx] = bytes([i])
        idx += 1
    return vocab

def merge_token(token_seq: tuple[bytes, ...], target: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    i = 0
    merged = []
    while i < len(token_seq):
        if i + 1 < len(token_seq) and token_seq[i] == target[0] and token_seq[i+1] == target[1]:
            merged.append(target[0] + target[1])
            i += 2
        else:
            merged.append(token_seq[i])
            i += 1
    return tuple(merged)

def update_token_seq_freq(token_seq_freqs: dict[tuple[bytes, ...], int], best_pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    new_token_seq_freqs = Counter()
    for token_seq, freq in token_seq_freqs.items():
        new_seq = merge_token(token_seq, best_pair)
        new_token_seq_freqs[new_seq] += freq
    return dict(new_token_seq_freqs)

def iter_pairs(token_seq: tuple[bytes, ...]):
    return zip(token_seq, token_seq[1:])

def contains_pair(token_seq: tuple[bytes, ...], target: tuple[bytes, bytes]) -> bool:
    return any(pair == target for pair in zip(token_seq, token_seq[1:]))

def apply_merge_incremental(token_seq_freqs: dict[tuple[bytes, ...], int], pair_freqs: dict[tuple[bytes, bytes], int], best_pair: tuple[bytes, bytes]) -> tuple[dict[tuple[bytes, ...], int], dict[tuple[bytes, bytes], int]]:
    affected = []
    new_token_seq_freqs = Counter(token_seq_freqs)
    new_pair_freqs = Counter(pair_freqs)

    for token_seq, freq in token_seq_freqs.items():
        if contains_pair(token_seq, best_pair):
            affected.append((token_seq, freq))

    for old_seq, freq in affected:
        new_seq = merge_token(old_seq, best_pair)

        for pair in iter_pairs(old_seq):
            new_pair_freqs[pair] -= freq
            if new_pair_freqs[pair] == 0:
                del new_pair_freqs[pair]

        new_token_seq_freqs[old_seq] -= freq
        if new_token_seq_freqs[old_seq] == 0:
            del new_token_seq_freqs[old_seq]

        new_token_seq_freqs[new_seq] += freq
        for pair in iter_pairs(new_seq):
            new_pair_freqs[pair] += freq

    return dict(new_token_seq_freqs), dict(new_pair_freqs)



def train_bpe(input_path: str, vocab_size: int, special_tokens:list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    docs = load_docs(input_path, special_tokens)
    vocab = init_vocab(special_tokens)
    merges = []
    token_seq_freqs = build_token_seq_freqs(docs)
    pair_freqs = count_pairs(token_seq_freqs)
    while len(vocab) < vocab_size:
        best_pair = get_max_pairs(pair_freqs)
        if best_pair is None:
            break
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token
        merges.append(best_pair)
        token_seq_freqs, pair_freqs = apply_merge_incremental(token_seq_freqs, pair_freqs, best_pair)

    return vocab, merges

