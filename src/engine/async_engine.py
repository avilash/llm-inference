import asyncio
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch

from src.engine.scheduler import EngineScheduler
from src.engine.sequence import SamplingParams, Sequence, SequenceStatus
from src.models import SimpleModelRunner, get_model_runner

logger = logging.getLogger(__name__)


def sample_tokens(
    logits: torch.Tensor,
    temperatures: List[float],
    top_ps: List[float],
) -> List[int]:
    """
    Sample next token IDs from last-token logits for each sequence in batch.
    Supports deterministic greedy decoding and stochastic nucleus (top-p) sampling.
    """
    batch_size = logits.shape[0]
    next_tokens: List[int] = []

    for i in range(batch_size):
        seq_logits = logits[i]
        temp = temperatures[i]
        top_p = top_ps[i]

        if temp <= 1e-4:
            token_id = int(torch.argmax(seq_logits).item())
        else:
            scaled_logits = seq_logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)

            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled_logits = scaled_logits.clone()
                scaled_logits[indices_to_remove] = -float("Inf")
                probs = torch.softmax(scaled_logits, dim=-1)

            token_id = int(torch.multinomial(probs, num_samples=1).item())

        next_tokens.append(token_id)

    return next_tokens


class AsyncInferenceEngine:
    """
    High-throughput asynchronous inference engine coordinating
    the continuous batching scheduler and GPU model execution.
    """

    def __init__(
        self,
        runner: Optional[SimpleModelRunner] = None,
        max_batch_size: int = 8,
    ):
        self.runner = runner or get_model_runner()
        self.scheduler = EngineScheduler(max_batch_size=max_batch_size)
        self._is_running: bool = False
        self._step_task: Optional[asyncio.Task] = None

    @property
    def model_name(self) -> str:
        """Name of the currently targeted model."""
        return self.runner.model_name

    def load_model(
        self,
        model_name: Optional[str] = None,
        force_reload: bool = False,
    ) -> None:
        """Preload model weights onto target hardware via model runner."""
        self.runner.load_model(model_name=model_name, force_reload=force_reload)

    def start(self) -> None:
        """Start the background continuous batching step loop."""
        if not self.runner.is_loaded:
            self.load_model()
        if self._is_running:
            return
        self._is_running = True
        self._step_task = asyncio.create_task(self._step_loop())
        logger.info("Continuous batching AsyncInferenceEngine started.")

    async def stop(self) -> None:
        """Gracefully stop the background step loop."""
        self._is_running = False
        if self._step_task and not self._step_task.done():
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass
        logger.info("Continuous batching AsyncInferenceEngine stopped.")

    async def _step_loop(self) -> None:
        """Main background loop driving iteration-level batch execution."""
        while self._is_running:
            try:
                if not self.scheduler.has_work():
                    await asyncio.sleep(0.005)
                    continue

                active_batch = self.scheduler.schedule()
                if not active_batch:
                    await asyncio.sleep(0.005)
                    continue

                await self._execute_iteration(active_batch)
                # Yield control back to asyncio event loop
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in continuous batching step loop: {e}")
                await asyncio.sleep(0.01)

    async def _execute_iteration(self, active_batch: List[Sequence]) -> None:
        """Execute a single token generation step across all active sequences."""
        batch_token_ids = [seq.all_token_ids for seq in active_batch]
        temperatures = [seq.sampling_params.temperature for seq in active_batch]
        top_ps = [seq.sampling_params.top_p for seq in active_batch]

        # Offload batched GPU forward pass to worker thread
        logits = await asyncio.to_thread(self.runner.forward_batch_logits, batch_token_ids)
        next_tokens = sample_tokens(logits, temperatures, top_ps)

        eos_token_id = self.runner.tokenizer.eos_token_id if self.runner.tokenizer else None

        for seq, token_id in zip(active_batch, next_tokens):
            is_stop = (token_id in seq.sampling_params.stop_token_ids) or (token_id == eos_token_id)

            if is_stop:
                seq.status = SequenceStatus.FINISHED
                seq.finish_reason = "stop"
                # Dispatch completion delta
                await seq.output_queue.put({
                    "text": "",
                    "finish_reason": "stop",
                    "num_prompt_tokens": seq.num_prompt_tokens,
                    "num_output_tokens": seq.num_output_tokens,
                })
                await seq.output_queue.put(None)
            else:
                # Calculate delta via prefix detokenization to correctly handle multi-byte
                # UTF-8 characters and BPE subword merges across token boundaries.
                prev_text = self.runner.decode(seq.output_token_ids)
                seq.append_token(token_id)
                curr_text = self.runner.decode(seq.output_token_ids)
                delta_text = curr_text[len(prev_text):]

                is_length_limit = seq.num_output_tokens >= seq.sampling_params.max_tokens
                finish_reason = "length" if is_length_limit else None

                if is_length_limit:
                    seq.status = SequenceStatus.FINISHED
                    seq.finish_reason = "length"

                await seq.output_queue.put({
                    "text": delta_text,
                    "finish_reason": finish_reason,
                    "num_prompt_tokens": seq.num_prompt_tokens,
                    "num_output_tokens": seq.num_output_tokens,
                })

                if is_length_limit:
                    await seq.output_queue.put(None)

    async def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Asynchronously stream generation results for a prompt."""
        params = sampling_params or SamplingParams()
        if not params.stop_token_ids and self.runner.tokenizer:
            params.stop_token_ids = {self.runner.tokenizer.eos_token_id}

        prompt_token_ids = self.runner.encode(prompt)
        seq_id = f"seq_{uuid.uuid4().hex[:8]}"
        seq = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=params,
        )

        self.scheduler.add_sequence(seq)

        try:
            while True:
                item = await seq.output_queue.get()
                if item is None:
                    break
                yield item
        except asyncio.CancelledError:
            self.scheduler.abort_sequence(seq_id)
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Asynchronously stream chat generation using model chat template."""
        prompt = self.runner.apply_chat_template(messages)
        async for chunk in self.generate(prompt=prompt, sampling_params=sampling_params):
            yield chunk


_engine: Optional[AsyncInferenceEngine] = None


def get_async_engine() -> AsyncInferenceEngine:
    """Return singleton instance of AsyncInferenceEngine."""
    global _engine
    if _engine is None:
        _engine = AsyncInferenceEngine()
    return _engine
