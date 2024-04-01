import torch
import statistics


@torch.no_grad()
def benchmark(model, input, num_iters=10):
    """Runs the model on the input several times and returns the median execution time."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(num_iters):
        start.record()
        model(input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return statistics.median(times)