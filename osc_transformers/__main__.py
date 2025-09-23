from random import randint
import time

from jsonargparse import auto_cli
from loguru import logger

from osc_transformers import TransformerDecoder, Sequence


def bench(
    cfg: str,
    num_seqs: int = 64,
    max_input_len: int = 1024,
    max_output_len: int = 1024,
):
    model = TransformerDecoder.form_config(config=cfg)
    max_model_len = max_input_len + max_output_len
    model.setup(
        max_model_len=max_model_len,
        num_kvcache_blocks=max_model_len * num_seqs // 256,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))]
        for _ in range(num_seqs)
    ]
    seqs = [
        Sequence(
            token_ids=prompt_token_ids[i],
            max_generate_tokens=max_output_len,
            ignore_eos=True,
        )
        for i in range(num_seqs)
    ]
    # warmup
    logger.info("warmup")
    _ = model.batch(seqs[:1])
    seqs[0].reset()
    # bench
    logger.info("start bench")
    start_time = time.perf_counter()
    seqs = model.batch(seqs)
    end_time = time.perf_counter()
    total_tokens = sum(seq.max_generate_tokens for seq in seqs)
    throughput = total_tokens / (end_time - start_time)
    print(f"Throughput: {throughput:.2f} tokens/s")


if __name__ == "__main__":
    auto_cli(bench)
