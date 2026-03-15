#!/usr/bin/env bash
set -e

uv run python cs336_basics/train_together.py \
  --train_data data/tinystories_train_tokens.bin \
  --val_data data/tinystories_valid_tokens.bin \
  --save_dir checkpoints/tinystories_exp7_2 \
  --log_dir runs/tinystories_exp7_2 \
  --vocab_size 10000 \
  --context_length 256 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 16 \
  --d_ff 1344 \
  --rope_theta 10000 \
  --batch_size 64 \
  --max_iters 20000 \
  --max_lr 3e-4 \
  --min_lr 3e-5 \
  --warmup_iters 500 \
  --cosine_iters 20000 \
  --weight_decay 0.1 \
  --beta1 0.9 \
  --beta2 0.95 \
  --eps 1e-8 \
  --max_grad_norm 1.0 \
  --eval_interval 200 \
  --eval_iters 20 \
  --log_interval 20 \
  --save_interval 1000 \
  --device cuda