import argparse
from cs336_basics.transformer_lm import TransformerLM
from train_transformer_lm import AdamWOpt, cross_entropy, gradient_clipping, learning_rate_schedule
import torch
from train_loop import data_loading, save_checkpoint, load_checkpoint
import numpy as np
import os
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def evaluate(model, val_data, batch_size, context_length, device, eval_iters):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = data_loading(val_data, batch_size, context_length, device)
            output = model(x)
            batch, context, vocab = output.shape
            output = output.reshape(batch * context, vocab)
            y = y.reshape(batch * context)
            loss = cross_entropy(output, y)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def get_grad_norm(parameters):
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0
    total = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    return total.item()

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data = np.memmap(args.train_data, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data, dtype=np.uint16, mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta,
        device = device
    ).to(device)

    optimizer = AdamWOpt(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    start_iter = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    progress_bar = tqdm(range(start_iter, args.max_iters), desc="Training", dynamic_ncols=True)

    for step in progress_bar:
        lr = learning_rate_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = data_loading(train_data, args.batch_size, args.context_length, device)

        # Forward
        output = model(x)
        batch, context, vocab = output.shape
        output = output.reshape(batch * context, vocab)
        y = y.reshape(batch * context)
        loss = cross_entropy(output, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm_before_clip = get_grad_norm(model.parameters())
        gradient_clipping(model.parameters(), args.max_grad_norm)
        grad_norm_after_clip = get_grad_norm(model.parameters())
        optimizer.step()

        progress_bar.set_postfix(train_loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

        writer.add_scalar("train/loss", loss.item(), step)
        writer.add_scalar("train/lr", lr, step)
        writer.add_scalar("train/grad_norm_before_clip", grad_norm_before_clip, step)
        writer.add_scalar("train/grad_norm_after_clip", grad_norm_after_clip, step)

        if step % args.log_interval == 0:
            print(f"[iter {step:6d}] train_loss={loss.item():.4f} lr={lr:.6e}")

        if step % args.eval_interval == 0:
            val_loss = evaluate(
                model=model,
                val_data=val_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                eval_iters=args.eval_iters,
            )
            print(f"[iter {step:6d}] val_loss={val_loss:.4f}")
            writer.add_scalar("val/loss", val_loss, step)

        if step % args.save_interval == 0 and step > start_iter:
            ckpt_path = save_dir / f"ckpt_step_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    final_ckpt = save_dir / "final_checkpoint.pt"
    save_checkpoint(model, optimizer, args.max_iters, final_ckpt)
    print(f"Training finished. Final checkpoint saved to {final_ckpt}")

    writer.close()

def build_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None)

    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # Optimization
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--cosine_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Eval and logging
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_dir", type=str, required=True)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)