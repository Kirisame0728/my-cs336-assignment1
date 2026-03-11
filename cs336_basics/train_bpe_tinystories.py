import os
import time
import json
import pickle
import tracemalloc
from pathlib import Path

from train_bpe import train_bpe


def bytes_to_safe_str(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("latin1")


def save_vocab_json(vocab: dict[int, bytes], out_path: Path) -> None:
    serializable_vocab = {str(k): bytes_to_safe_str(v) for k, v in vocab.items()}
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)


def save_merges_txt(merges: list[tuple[bytes, bytes]], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for left, right in merges:
            left_s = bytes_to_safe_str(left)
            right_s = bytes_to_safe_str(right)
            f.write(f"{left_s}\t{right_s}\n")


def save_pickle(obj, out_path: Path) -> None:
    with out_path.open("wb") as f:
        pickle.dump(obj, f)


def find_longest_token(vocab: dict[int, bytes]) -> tuple[int, bytes]:
    token_id = max(vocab, key=lambda k: len(vocab[k]))
    return token_id, vocab[token_id]


def format_size_mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def main():
    input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]

    out_dir = Path("data/tinystories_bpe")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print("Starting BPE training on TinyStories...")
    print(f"Input path     : {input_path}")
    print(f"Vocab size     : {vocab_size}")
    print(f"Special tokens : {special_tokens}")
    print(f"Output dir     : {out_dir}")
    print()

    tracemalloc.start()
    t0 = time.perf_counter()

    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    elapsed = time.perf_counter() - t0
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    longest_token_id, longest_token_bytes = find_longest_token(vocab)
    longest_token_len = len(longest_token_bytes)

    try:
        longest_token_text = longest_token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        longest_token_text = longest_token_bytes.decode("utf-8", errors="replace")

    expected_merges = vocab_size - 256 - len(special_tokens)

    save_vocab_json(vocab, out_dir / "vocab.json")
    save_merges_txt(merges, out_dir / "merges.txt")
    save_pickle(vocab, out_dir / "vocab.pkl")
    save_pickle(merges, out_dir / "merges.pkl")

    summary = {
        "input_path": str(input_path),
        "vocab_size_requested": vocab_size,
        "vocab_size_actual": len(vocab),
        "num_merges": len(merges),
        "expected_num_merges_if_full": expected_merges,
        "special_tokens": special_tokens,
        "elapsed_seconds": elapsed,
        "peak_memory_mb_tracemalloc": format_size_mb(peak_mem),
        "longest_token_id": longest_token_id,
        "longest_token_num_bytes": longest_token_len,
        "longest_token_text_preview": longest_token_text[:200],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print(f"Actual vocab size      : {len(vocab)}")
    print(f"Number of merges       : {len(merges)}")
    print(f"Expected full merges   : {expected_merges}")
    print(f"Elapsed time (seconds) : {elapsed:.3f}")
    print(f"Peak memory (MB)       : {format_size_mb(peak_mem):.2f}")
    print()
    print(f"Longest token id       : {longest_token_id}")
    print(f"Longest token bytes    : {longest_token_len}")
    print(f"Longest token preview  : {repr(longest_token_text[:200])}")
    print()
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()

