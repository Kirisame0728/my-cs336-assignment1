#!/usr/bin/env bash
set -e

uv run python cs336_basics/encode_tinystories_to_bin.py \
  --train_txt data/TinyStoriesV2-GPT4-train.txt \
  --valid_txt data/TinyStoriesV2-GPT4-valid.txt \
  --vocab_pkl data/tinystories_bpe/vocab.pkl \
  --merges_pkl data/tinystories_bpe/merges.pkl \
  --train_bin data/tinystories_train_tokens.bin \
  --valid_bin data/tinystories_valid_tokens.bin