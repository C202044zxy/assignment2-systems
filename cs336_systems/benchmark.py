from cs336_basics import *
import torch
from torch import Tensor
import timeit
import argparse


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
):
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    for _ in range(warmup_iter):
        xb, yb = random_batch(batch_size, context_length, vocab_size)
        logits = model(xb)
        loss = cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Explicitly delete intermediate tensors to free memory
        del logits, loss, xb, yb

    torch.cuda.synchronize()
    torch.cuda.empty_cache()  # Clear cache after warmup
    
    forward_time = 0
    backward_time = 0
    for _ in range(prof_iter):
        xb, yb = random_batch(batch_size, context_length, vocab_size)

        forward_start = timeit.default_timer()
        logits = model(xb)
        torch.cuda.synchronize()
        forward_end = timeit.default_timer()
        forward_time += forward_end - forward_start

        loss = cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)

        backward_start = timeit.default_timer()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        backward_end = timeit.default_timer()
        backward_time += backward_end - backward_start
        
        # Clean up intermediate tensors
        del logits, loss, xb, yb

    forward_time /= prof_iter
    backward_time /= prof_iter
    print(f"Forward elapsed time: {forward_time:.6f}")
    print(f"Backward elapsed time: {backward_time:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--warmup_iter", type=int, default=5)
    parser.add_argument("--prof_iter", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = BasicsTransformerLM(
        args.vocab_size, args.context_length, args.d_model, args.d_ff, args.num_layers, args.num_heads, 10000.0
    )
    model.to(device)
    optimizer = AdamW(model.parameters())
    print("model and optimizer initialized")

    benchmark(model, optimizer, args.warmup_iter, args.prof_iter, args.batch_size, args.context_length, args.vocab_size)


if __name__ == "__main__":
    main()
