from cs336_basics import *
import torch
from torch import Tensor
import timeit
import argparse
from contextlib import nullcontext

def random_batch(batch_size: int, context_length: int, vocab_size: int) -> tuple[Tensor, Tensor]:
    xb = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=torch.device("cuda"))
    yb = torch.randint(low=0, high=vocab_size, size=(batch_size, context_length), device=torch.device("cuda"))
    return xb, yb


def benchmark(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    warmup_iter: int,
    prof_iter: int,
    batch_size: int,
    context_length: int,
    vocab_size: int,
    amp: bool,
    mem_prof: bool,
):
    model.train()
    xb, yb = random_batch(batch_size, context_length, vocab_size)
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if amp
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp)

    for _ in range(warmup_iter):
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(xb)
            loss = cross_entropy(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()

    if mem_prof:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    forward_time = 0
    backward_time = 0
    for _ in range(prof_iter):
        optimizer.zero_grad(set_to_none=True)
        forward_start = timeit.default_timer()
        with amp_ctx:
            logits = model(xb)
            loss = cross_entropy(logits, yb)
        torch.cuda.synchronize()
        forward_time += timeit.default_timer() - forward_start

        backward_start = timeit.default_timer()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        backward_time += timeit.default_timer() - backward_start

    forward_time /= prof_iter
    backward_time /= prof_iter
    print(f"Forward elapsed time: {forward_time:.6f}")
    print(f"Backward elapsed time: {backward_time:.6f}")

    if mem_prof:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--warmup_iter", type=int, default=5)
    parser.add_argument("--prof_iter", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", help="Enable BF16 mixed precision")
    parser.add_argument("--mem_prof", action="store_true", help="Enable memory profiling")
    args = parser.parse_args()

    device = torch.device("cuda")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
    )
    model.to(device)
    optimizer = AdamW(model.parameters())
    print("model and optimizer initialized")

    benchmark(
        model=model,
        optimizer=optimizer,
        warmup_iter=args.warmup_iter,
        prof_iter=args.prof_iter,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        amp=args.amp,
        mem_prof=args.mem_prof
    )


if __name__ == "__main__":
    main()
