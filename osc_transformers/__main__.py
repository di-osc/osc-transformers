from random import randint
import time
from pathlib import Path

from jsonargparse import auto_cli
from loguru import logger

from osc_transformers import TransformerDecoder, Sequence, SamplingParams


def bench(
    cfg: str,
    num_seqs: int = 64,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
    gpu_memory_utilization: float = 0.9,
):
    """bench transformer decoder

    Args:
        cfg (str): the path of config file
        num_seqs (int, optional): number of sequences. Defaults to 64.
        max_input_len (int, optional): max input length. Defaults to 1024.
        max_output_len (int, optional): max output length. Defaults to 1024.
    """
    if not Path(cfg).exists():
        raise FileNotFoundError(f"Config file {cfg} not found")
    model = TransformerDecoder.from_config(config=cfg)
    max_model_len = max_input_len + max_output_len
    model.setup(
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    def create_seqs(num_seqs: int) -> list[Sequence]:
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(100, max_input_len))]
            for _ in range(num_seqs)
        ]
        seqs = [
            Sequence(
                token_ids=prompt_token_ids[i],
                sampling_params=SamplingParams(max_generate_tokens=max_output_len),
                ignore_eos=True,
            )
            for i in range(num_seqs)
        ]
        return seqs

    # warmup
    logger.info("🔥 Warming up the model")
    _ = model.batch(create_seqs(1))
    # bench 1 seq first token
    logger.info("📊 Starting benchmark for single sequence")
    seq = create_seqs(num_seqs=1)[0]
    start_time = time.perf_counter()
    first_response_time = None
    num_tokens = 0
    for _ in model.stream(seq=seq):
        if first_response_time is None:
            first_response_time = time.perf_counter() - start_time
        num_tokens += 1
    total_time = time.perf_counter() - start_time
    logger.success(
        f"🎯 Prompt tokens: {len(seq.prompt_token_ids)}, First token latency: {first_response_time:.3f}s, Tokens per second: {num_tokens / total_time:.1f} tokens/s"
    )
    # bench throughput
    logger.info("📊 Starting benchmark for batch inference")
    seqs = create_seqs(num_seqs=num_seqs)
    start_time = time.perf_counter()
    seqs = model.batch(seqs=seqs)
    end_time = time.perf_counter()
    total_tokens = sum(seq.num_completion_tokens for seq in seqs)
    throughput = total_tokens / (end_time - start_time)
    logger.success(f"🎯 Throughput: {throughput:.2f} tokens/s")


def run_cli():
    components = {"bench": bench}
    auto_cli(components)


if __name__ == "__main__":
    run_cli()
