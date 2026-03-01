from cs336_basics import scaled_dot_product_attention
import torch
import timeit

def attention_benchmark():
    d_models = [16, 32, 64, 128]
    sequence_lengths = [256, 1024, 4096]
    batch_size = 8
    warmup_iter = 10
    benchmark_iter = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmark running on {device}")

    for d_model in d_models:
        for sequence_length in sequence_lengths:
            print(f"\nBenchmarking d_model={d_model}, sequence_length={sequence_length}")

            Q = torch.randn(batch_size, sequence_length, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, sequence_length, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, sequence_length, d_model, device=device, requires_grad=True)
            
            # warm-up
            for _ in range(warmup_iter):
                out = scaled_dot_product_attention(Q, K, V)
            torch.cuda.synchronize()

            # forward
            forward_time = 0
            for _ in range(benchmark_iter):
                forward_start = timeit.default_timer()
                out = scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                forward_time += timeit.default_timer() - forward_start
            forward_time /= benchmark_iter
            print(f"Forward elapsed time {forward_time:.6f}")

            # backward
            backward_time = 0
            for _ in range(benchmark_iter):
                out = scaled_dot_product_attention(Q, K, V)
                out_grad = torch.randn_like(out)
                torch.cuda.synchronize()
                backward_start = timeit.default_timer()
                out.backward(out_grad, retain_graph=True)
                torch.cuda.synchronize()
                backward_time += timeit.default_timer() - backward_start
                Q.grad.zero_()
                K.grad.zero_()
                V.grad.zero_()
            backward_time /= benchmark_iter
            print(f"backward elapsed time {backward_time:.6f}")

if __name__ == "__main__":
    attention_benchmark()