from collections import deque
from queue import Queue
from typing import List

from loguru import logger

from .sequence import Sequence, SequenceStatus
from .block_manager import BlockManager


class Scheduler:
    def __init__(
        self,
        max_num_seqs: int,
        max_num_batched_tokens: int,
        eos: int,
        num_kvcache_blocks: int,
        kvcache_block_size: int,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos
        self.block_manager = BlockManager(
            num_blocks=num_kvcache_blocks, block_size=kvcache_block_size
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.response_queues: dict[int, Queue] = {}

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence, response_queue: Queue = None) -> None:
        assert (
            seq.status == SequenceStatus.WAITING
        ), f"new seq must be waiting, but got {seq.status}"
        self.waiting.append(seq)
        assert (
            seq.seq_id not in self.response_queues
        ), "seq {} already in response_queues".format(seq.seq_id)
        if response_queue is not None:
            self.response_queues[seq.seq_id] = response_queue
        else:
            self.response_queues[seq.seq_id] = Queue()

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
                logger.warning(
                    "batched tokens exceed max_num_batched_tokens for seq {} at prefill".format(
                        seq.seq_id
                    )
                )
                break
            if not self.block_manager.can_allocate(seq):
                logger.warning(
                    "can not allocate block for seq {} at prefill".format(seq.seq_id)
                )
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
            return scheduled_seqs, False
        return scheduled_seqs, None

    def preempt(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def check_finished(self, seqs: List[Sequence]) -> None:
        for seq in seqs:
            # 检查序列是否已经在 response_queues 中
            # 如果不在，说明已经被处理过了，跳过
            if seq.seq_id not in self.response_queues:
                continue
            response_queue = self.response_queues[seq.seq_id]
            if seq.stream_response:
                response_queue.put(seq.last_token)
            if (
                not seq.ignore_eos and seq.last_token == self.eos
            ) or seq.num_completion_tokens == seq.sampling_params.max_generate_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                del self.response_queues[seq.seq_id]
                if seq.stream_response:
                    response_queue.put(seq.end_char)
                else:
                    response_queue.put(seq)

    def set_all_failed(self, error_message: str) -> None:
        while self.waiting:
            self.set_failed(self.waiting.popleft(), error_message)
        while self.running:
            self.set_failed(self.running.popleft(), error_message)

    def set_failed(self, seq: Sequence, error_message: str) -> None:
        assert seq.status in [
            SequenceStatus.WAITING,
            SequenceStatus.RUNNING,
        ], f"seq {seq.seq_id} is not in waiting or running"
        if seq.status == SequenceStatus.WAITING:
            self.waiting.remove(seq)
        elif seq.status == SequenceStatus.RUNNING:
            self.running.remove(seq)
        seq.status = SequenceStatus.FAILED
        seq.error_message = error_message
        if seq.seq_id in self.response_queues:
            response_queue = self.response_queues[seq.seq_id]
            if seq.stream_response:
                response_queue.put(seq.end_char)
            else:
                response_queue.put(seq)
            del self.response_queues[seq.seq_id]
