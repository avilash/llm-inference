import logging
from collections import deque
from typing import List, Optional

from src.engine.sequence import Sequence, SequenceStatus

logger = logging.getLogger(__name__)


class EngineScheduler:
    """
    Continuous (iteration-level) scheduler that dynamically manages
    active sequences in the batch and queued waiting sequences.
    """

    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        self.waiting_queue: deque[Sequence] = deque()
        self.running_batch: List[Sequence] = []

    def add_sequence(self, sequence: Sequence) -> None:
        """Add newly arrived sequence to waiting queue."""
        sequence.status = SequenceStatus.WAITING
        self.waiting_queue.append(sequence)
        logger.debug(f"Enqueued sequence {sequence.seq_id}. Queue size: {len(self.waiting_queue)}")

    def abort_sequence(self, seq_id: str) -> None:
        """Cancel an ongoing or waiting sequence."""
        # Check waiting queue
        for seq in list(self.waiting_queue):
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "cancelled"
                seq.output_queue.put_nowait(None)
                self.waiting_queue.remove(seq)
                logger.info(f"Aborted waiting sequence {seq_id}")
                return

        # Check running batch
        for seq in self.running_batch:
            if seq.seq_id == seq_id:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "cancelled"
                seq.output_queue.put_nowait(None)
                logger.info(f"Aborted running sequence {seq_id}")
                return

    def has_work(self) -> bool:
        """Check if there are any sequences to process."""
        return bool(self.running_batch or self.waiting_queue)

    def schedule(self) -> List[Sequence]:
        """
        Iteration-level scheduling step:
        1. Evict any sequences that completed in the previous step.
        2. Admit waiting sequences into available batch slots.
        3. Return currently active batch.
        """
        # 1. Evict finished sequences
        self.running_batch = [seq for seq in self.running_batch if not seq.is_finished()]

        # 2. Admit new sequences from waiting queue if capacity allows
        while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
            new_seq = self.waiting_queue.popleft()
            new_seq.status = SequenceStatus.RUNNING
            self.running_batch.append(new_seq)
            logger.debug(f"Admitted sequence {new_seq.seq_id} to active batch (size {len(self.running_batch)})")

        return list(self.running_batch)
