from collections import deque

import xxhash

from .sequence import Sequence


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def reset(self):
        for block in self.blocks:
            block.reset()
        self.hash_to_block_id.clear()
        self.free_block_ids.clear()
        self.used_block_ids.clear()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(b"".join(token_id.to_bytes(8, "little", signed=True) for token_id in token_ids))
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if there are enough free blocks to allocate for a sequence.

        This method determines whether the block manager has sufficient free blocks
        to satisfy the memory requirements of the given sequence. It's a lightweight
        check that compares the number of available free blocks against the number
        of blocks needed by the sequence.

        Args:
            seq (Sequence): The sequence to check allocation feasibility for. The
                          sequence must have num_blocks attribute set indicating
                          how many blocks it requires.

        Returns:
            bool: Returns True if there are enough free blocks (len(free_block_ids) >= seq.num_blocks),
                 False otherwise.

        Examples:
            >>> # Sequence requiring 3 blocks
            >>> seq.num_blocks  # 3
            >>> manager.free_block_ids  # [1, 5, 7, 9]
            >>> len(manager.free_block_ids)  # 4
            >>> manager.can_allocate(seq)  # True (4 >= 3)

            >>> # Sequence requiring 5 blocks but only 2 available
            >>> seq.num_blocks  # 5
            >>> manager.free_block_ids  # [2, 8]
            >>> len(manager.free_block_ids)  # 2
            >>> manager.can_allocate(seq)  # False (2 < 5)

        Note:
            - This is a read-only check that doesn't modify any state
            - Used before calling allocate() to ensure allocation will succeed
            - Considers only currently free blocks, not cached reusable blocks
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Allocate blocks for a sequence with intelligent caching and reuse.

        This method allocates memory blocks for a sequence, implementing an advanced
        caching strategy that reuses existing blocks when possible. It computes hashes
        for complete blocks and checks if identical blocks already exist in memory.
        When cache hits occur, reference counts are incremented instead of allocating
        new blocks, enabling efficient memory sharing across sequences.

        The allocation process:
        1. Iterates through sequence blocks in order
        2. Computes rolling hash for each complete block
        3. Checks hash_to_block_id mapping for existing blocks
        4. Reuses cached blocks (increments ref_count) or allocates new ones
        5. Updates hash mappings for future reuse

        Args:
            seq (Sequence): The sequence to allocate blocks for. Must have:
                          - Empty block_table (fresh allocation)
                          - num_blocks indicating required blocks
                          - block(i) method to access individual blocks

        Examples:
            >>> # Fresh sequence with no cached blocks available
            >>> seq.block_table  # []
            >>> seq.num_blocks  # 2
            >>> manager.free_block_ids  # [1, 3, 5]
            >>> manager.hash_to_block_id  # {} (empty cache)
            >>> manager.allocate(seq)
            >>> # Allocates blocks 1 and 3, seq.block_table becomes [1, 3]

            >>> # Sequence with some blocks available in cache
            >>> seq.num_blocks  # 3
            >>> manager.hash_to_block_id  # {hash_A: 2, hash_B: 4}
            >>> # Block 0 matches hash_A, block 2 matches hash_B
            >>> manager.allocate(seq)
            >>> # Reuses blocks 2 and 4 (ref_count += 1), allocates 1 new block
            >>> seq.block_table  # [2, new_block, 4]
            >>> seq.num_cached_tokens  # 32 (2 cached blocks * 16 tokens each)

        Note:
            - Requires seq.block_table to be empty (asserted)
            - Implements rolling hash for prefix-aware caching
            - Cache misses trigger new block allocation
            - Cache hits increment reference counts for memory sharing
            - Updates num_cached_tokens to track computation reuse
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Deallocate all blocks associated with a sequence.

        This method releases all memory blocks that were allocated for the given sequence.
        It iterates through the sequence's block table in reverse order, decrements the reference
        count for each block, and physically deallocates blocks when their reference count reaches zero.
        Finally, it resets the sequence's cached token count and clears its block table.

        The deallocation process is reference-counted, meaning blocks are only physically freed
        when no other sequences are using them (ref_count becomes 0). This allows for efficient
        memory sharing across multiple sequences that may reference the same blocks.

        Args:
            seq (Sequence): The sequence whose blocks should be deallocated. This sequence
                          object contains the block_table that tracks all allocated blocks
                          and num_cached_tokens that tracks cached computation state.

        Examples:
            >>> # Sequence with blocks [1, 2, 3] and ref_counts [2, 1, 3]
            >>> seq.block_table  # [1, 2, 3]
            >>> seq.num_cached_tokens  # 45
            >>> manager.blocks[1].ref_count  # 2
            >>> manager.blocks[2].ref_count  # 1
            >>> manager.blocks[3].ref_count  # 3
            >>> manager.deallocate(seq)
            >>> # Block 2 gets deallocated (ref_count becomes 0)
            >>> # Blocks 1 and 3 ref_counts become 1 and 2 respectively
            >>> seq.block_table  # []
            >>> seq.num_cached_tokens  # 0

        Note:
            - Blocks are processed in reverse order to maintain proper deallocation sequence
            - Only blocks with ref_count == 0 after decrementing are physically deallocated
            - The sequence's block_table is completely cleared after deallocation
            - This operation is typically called when a sequence is finished or needs to be reset
        """
        for block_id in reversed(seq.block_table):
            block: Block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check if the sequence can continue to expand.

        This method checks if the system has enough free blocks available when the sequence
        needs to allocate a new block. Specifically, when the sequence length modulo block size
        equals 1, it means the next token will occupy the first position of a new block,
        and sufficient free blocks must be available.

        The method makes its decision by comparing the number of free blocks (len(self.free_block_ids))
        against whether a new block is needed. free_block_ids is a list maintained by BlockManager
        containing all currently unallocated block IDs.

        Args:
            seq (Sequence): The sequence object to check. This sequence contains current
                          block allocation information and sequence length properties.

        Returns:
            bool: Returns True if the sequence can continue to expand (either the current
                 block has space or sufficient free blocks are available); returns False
                 if a new block is needed but free blocks are insufficient.

        Examples:
            Assuming block_size is 16:

            >>> # Sequence length is 15, current block has 1 position left
            >>> len(seq)  # 15
            >>> 15 % 16 == 1  # False, no new block needed
            >>> len(manager.free_block_ids)  # Number of free blocks doesn't matter
            >>> manager.can_append(seq)  # Returns True (current block has space)
            True

            >>> # Sequence length is 16, current block is exactly full
            >>> len(seq)  # 16
            >>> 16 % 16 == 0  # False, no new block needed (next token can still be added)
            >>> len(manager.free_block_ids)  # Number of free blocks doesn't matter
            >>> manager.can_append(seq)  # Returns True
            True

            >>> # Sequence length is 17, next token needs a new block
            >>> len(seq)  # 17
            >>> 17 % 16 == 1  # True, new block needed
            >>> manager.free_block_ids  # [5, 12, 8] (has 3 free blocks available)
            >>> len(manager.free_block_ids) >= 1  # 3 >= 1, check passes
            >>> manager.can_append(seq)  # Returns True since free blocks are available
            True

            >>> # Sequence length is 17, but no free blocks available
            >>> len(seq)  # 17
            >>> 17 % 16 == 1  # True, new block needed
            >>> manager.free_block_ids  # [] (no free blocks available)
            >>> len(manager.free_block_ids) >= 1  # 0 >= 1, check fails
            >>> manager.can_append(seq)  # Returns False due to insufficient free blocks
            False

        Note:
            This check is based on the following logic:
            - When len(seq) % block_size == 1, the next token needs a new block, check for at least 1 free block
            - When len(seq) % block_size != 1, current block has space, no new block needed, returns True directly
            This design ensures resource checking only when truly needed, avoiding unnecessary restrictions.
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Prepare a sequence for appending by managing block allocation and hashing.

        This method handles the block-level preparation when a sequence needs to expand.
        It manages three scenarios based on the sequence length modulo block size:
        - When len(seq) % block_size == 1: Allocates a new block for the next token
        - When len(seq) % block_size == 0: Updates the hash of the current block for caching
        - Otherwise: Prepares the current block for continued token addition

        The method implements a paged attention optimization strategy where blocks are
        hashed and cached to enable efficient memory sharing and computation reuse across
        sequences with identical prefixes.

        Args:
            seq (Sequence): The sequence that may need block expansion. The sequence must
                          have a valid block_table and the last block should be properly
                          initialized for the current operation.

        Examples:
            >>> # Case 1: Sequence length 16, block_size 16, need new block
            >>> len(seq)  # 16
            >>> 16 % 16  # 0, but wait for next token
            >>> # Actually triggered when sequence becomes length 17
            >>> len(seq)  # 17
            >>> 17 % 16  # 1, allocate new block
            >>> manager.free_block_ids  # [5, 12, 8]
            >>> manager.may_append(seq)
            >>> # Block 5 gets allocated and added to seq.block_table

            >>> # Case 2: Sequence length exactly divisible by block_size
            >>> len(seq)  # 32
            >>> 32 % 16  # 0, update current block hash
            >>> manager.may_append(seq)
            >>> # Current block gets hashed and registered for future reuse

        Note:
            - This method assumes the sequence is in a valid state for expansion
            - Block allocation only occurs when len(seq) % block_size == 1
            - Hash computation enables prefix caching and memory sharing
            - The method modifies the sequence's block_table and block states
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks - 1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
