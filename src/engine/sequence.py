import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SequenceStatus(str, Enum):
    """Lifecycle status of a sequence in the inference engine."""
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"


@dataclass
class SamplingParams:
    """Parameters governing token sampling and stopping conditions."""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 128
    stop_token_ids: Set[int] = field(default_factory=set)


@dataclass
class Sequence:
    """Represents a single generation request and its progressive state."""
    seq_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: Optional[str] = None
    output_queue: asyncio.Queue[Optional[Dict[str, Any]]] = field(default_factory=asyncio.Queue)

    @property
    def all_token_ids(self) -> List[int]:
        """Return combined prompt and generated tokens."""
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    def append_token(self, token_id: int) -> None:
        """Append newly generated token to sequence."""
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
