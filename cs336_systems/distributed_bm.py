import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def warm_up(warm_up_iter: int = 5):
    for _ in range(warm_up_iter):
        data = torch.randint(0, 10, (3,))
        dist.all_reduce(data, async_op=False)

def benchmark(benchmark_iter: int = 100):
    avg_time = 0
    for _ in range(benchmark_iter):
        data = torch.randint(0, 10, (3,))
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        end_time = timeit.default_timer()
        avg_time += end_time - start_time
    avg_time /= benchmark_iter
    return avg_time

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    
    warm_up()
    local_result = benchmark()

    local_result = torch.tensor([local_result])
    all_results = [torch.zeros(1) for _ in range(world_size)]
    dist.all_gather(all_results, local_result)

    if rank == 0:
        avg_time = 0
        for i, result in enumerate(all_results):
            avg_time += result.item()
        avg_time /= world_size
        print(f"Average time: {avg_time:.6f}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)