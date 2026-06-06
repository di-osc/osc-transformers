import torch

from osc_transformers.block_manager import BlockManager
from osc_transformers.sequence import Sequence


def test_allocate_reuses_token_prefix_cache_without_prompt_embeds():
    manager = BlockManager(num_blocks=4, block_size=2)
    first = Sequence([1, 2], block_size=2)
    second = Sequence([1, 2], block_size=2)

    manager.allocate(first)
    manager.allocate(second)

    assert second.num_cached_tokens == 2
    assert second.block_table == first.block_table


def test_allocate_does_not_reuse_token_prefix_cache_for_prompt_embeds():
    manager = BlockManager(num_blocks=4, block_size=2)
    first = Sequence([1, 2], prompt_embeds=torch.zeros(2, 4), block_size=2)
    second = Sequence([1, 2], prompt_embeds=torch.ones(2, 4), block_size=2)

    manager.allocate(first)
    manager.allocate(second)

    assert second.num_cached_tokens == 0
    assert second.block_table != first.block_table
    assert all(block.hash == -1 for block in manager.blocks if block.ref_count)


def test_may_append_does_not_require_hashes_for_prompt_embeds():
    manager = BlockManager(num_blocks=4, block_size=2)
    seq = Sequence([1, 2], prompt_embeds=torch.zeros(2, 4), block_size=2)
    manager.allocate(seq)
    assert manager.blocks[seq.block_table[-1]].hash == -1
    seq.append_token(3)

    manager.may_append(seq)

    assert len(seq.block_table) == 2
