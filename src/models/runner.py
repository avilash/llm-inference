import logging
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class SimpleModelRunner:
    """
    Model runner managing the underlying PyTorch model and tokenizer on target hardware.
    Provides tokenization utilities, batched single-step forward passes for continuous batching,
    and legacy baseline synchronous generation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        settings = get_settings()
        self.model_name = model_name or settings.model_name

        target_device = device or settings.model_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target_device)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.device.type == "cuda":
            self.dtype = dtype_map.get(settings.model_dtype.lower(), torch.float16)
        else:
            self.dtype = torch.float32

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.is_loaded: bool = False

    # -------------------------------------------------------------------------
    # Lifecycle & Model Loading
    # -------------------------------------------------------------------------

    def load_model(
        self,
        model_name: Optional[str] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Load model weights & tokenizer into device memory.

        Args:
            model_name: Optional model identifier/path. If provided and different
                        from current model_name, updates target model and forces reload.
            force_reload: If True, forces reloading even if already loaded.
        """
        if model_name and model_name != self.model_name:
            self.model_name = model_name
            force_reload = True

        if self.is_loaded and not force_reload and self.model is not None and self.tokenizer is not None:
            logger.info(f"Model {self.model_name} is already loaded on {self.device}.")
            return

        logger.info(f"Loading {self.model_name} on {self.device} ({self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.is_loaded = True
        logger.info(f"Model {self.model_name} loaded successfully on {self.device}.")

    # -------------------------------------------------------------------------
    # Tokenization & Chat Helpers
    # -------------------------------------------------------------------------

    def encode(self, prompt: str) -> List[int]:
        """Convert text prompt to token IDs."""
        if not self.is_loaded or self.tokenizer is None:
            self.load_model()
        assert self.tokenizer is not None
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not self.is_loaded or self.tokenizer is None:
            self.load_model()
        assert self.tokenizer is not None
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply model-native chat template to message list."""
        if not self.is_loaded or self.tokenizer is None:
            self.load_model()
        assert self.tokenizer is not None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # -------------------------------------------------------------------------
    # Batched Forward Execution (Core Primitive for Continuous Batching)
    # -------------------------------------------------------------------------

    def forward_batch_logits(self, batch_token_ids: List[List[int]]) -> torch.Tensor:
        """
        Execute a single batched forward pass for active sequences in continuous batching.
        Uses left-padding so the final position is always the last active token.

        Args:
            batch_token_ids: List of token ID sequences (one list per active sequence).

        Returns:
            Last-position logits tensor of shape [batch_size, vocab_size].
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            self.load_model()
        assert self.model is not None and self.tokenizer is not None

        batch_size = len(batch_token_ids)
        if batch_size == 0:
            raise ValueError("Cannot run forward pass on empty batch.")

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        max_len = max(len(seq) for seq in batch_token_ids)

        # Build left-padded input_ids and attention_mask tensors
        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=self.device)

        for i, seq in enumerate(batch_token_ids):
            seq_len = len(seq)
            # Left padding ensures last token (-1) is the actual token predicting the next token
            input_ids[i, max_len - seq_len :] = torch.tensor(seq, dtype=torch.long, device=self.device)
            attention_mask[i, max_len - seq_len :] = 1

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        # Logits for the final position across the batch: [batch_size, vocab_size]
        return outputs.logits[:, -1, :]


# Global singleton instance
_runner: Optional[SimpleModelRunner] = None


def get_model_runner() -> SimpleModelRunner:
    """Return cached singleton instance of SimpleModelRunner."""
    global _runner
    if _runner is None:
        _runner = SimpleModelRunner()
    return _runner
