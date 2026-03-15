import argparse
import pickle
from pathlib import Path

import numpy as np

from cs336_basics.train_bpe import Tokenizer


def load_tokenizer(vocab_pkl: str, merges_pkl: str, special_tokens: list[str]):
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    return tokenizer, vocab, merges


def encode_text_file(tokenizer, txt_path: str):
    text = Path(txt_path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    return ids


def save_bin(ids, out_path: str):
    arr = np.asarray(ids, dtype=np.uint16)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(out_path)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", type=str, required=True)
    parser.add_argument("--valid_txt", type=str, required=True)
    parser.add_argument("--vocab_pkl", type=str, required=True)
    parser.add_argument("--merges_pkl", type=str, required=True)
    parser.add_argument("--train_bin", type=str, required=True)
    parser.add_argument("--valid_bin", type=str, required=True)
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    tokenizer, vocab, merges = load_tokenizer(
        args.vocab_pkl,
        args.merges_pkl,
        special_tokens
    )

    print("Loaded tokenizer.")
    print(f"Vocab size: {len(vocab)}")
    print(f"Num merges: {len(merges)}")

    train_ids = encode_text_file(tokenizer, args.train_txt)
    train_arr = save_bin(train_ids, args.train_bin)
    print(f"Train tokens: {len(train_arr)}")
    print(f"Saved train bin to: {args.train_bin}")

    valid_ids = encode_text_file(tokenizer, args.valid_txt)
    valid_arr = save_bin(valid_ids, args.valid_bin)
    print(f"Valid tokens: {len(valid_arr)}")
    print(f"Saved valid bin to: {args.valid_bin}")

    if train_arr.max() >= len(vocab):
        raise ValueError("Train token id exceeds vocab size.")
    if valid_arr.max() >= len(vocab):
        raise ValueError("Valid token id exceeds vocab size.")

    print("Done.")


if __name__ == "__main__":
    main()