"""
Generator wrapper: Qwen2.5-3B-Instruct, 4-bit, greedy, batched.

Held constant across all 16 runs: one prompt, temperature 0, 32 new tokens, the
same abstention token. The only thing that varies between runs is the context
the retriever hands over -- which is the point.

Hardware notes for a 6 GB Turing card (GTX 1660 Ti), all measured on that
card rather than assumed:
  * **fp32 compute, not fp16.** 0.12 vs 0.04 prompts/s -- TU116 has no tensor
    cores and the fp16 GEMM path is a slow fallback, exactly as for the
    embedders.
  * SDPA attention, never FlashAttention-2 -- that needs Ampere or newer.
  * 4-bit NF4 weights are ~2 GB, and GQA keeps the KV cache small, but the
    prefill logits tensor is not small: see _generate_kwargs. Without capping
    it, batch 4 peaked at 9 GB on a 6 GB card and ran slower than batch 1.
"""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

NO_ANSWER = "NO_ANSWER"


def transformers_major() -> int:
    """Major version of the installed transformers, or 0 if absent.

    v5 renamed `torch_dtype` to `dtype` on from_pretrained. The pinned-version
    approach would be to require v4; the portable one is to ask.
    """
    try:
        import transformers
        return int(str(transformers.__version__).split(".")[0])
    except Exception:
        return 0


_CLEAN_PREFIX = re.compile(r"^\s*(answer|a)\s*[:\-]\s*", re.IGNORECASE)


def clean_answer(raw: str, max_chars: int = 200) -> str:
    """Trim a chat model's output down to the span we asked for.

    Deliberately conservative: strip an echoed "Answer:", take the first line,
    drop wrapping quotes. Nothing here tries to *find* an answer inside a
    verbose reply -- that would be scoring our own post-processing instead of
    the model, and it would interact with chunking in ways we could not
    untangle.
    """
    if not raw:
        return ""
    text = raw.strip().split("\n")[0].strip()
    text = _CLEAN_PREFIX.sub("", text).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text[:max_chars]


class BaseGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompts: Sequence[str], batch_size: Optional[int] = None) -> List[str]:
        ...

    def describe(self) -> Dict[str, Any]:
        return {"generator": self.name}


class QwenGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        max_new_tokens: int = 32,
        load_in_4bit: bool = True,
        device: Optional[str] = None,
        batch_size: int = 8,
        dtype: str = "float16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype)

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self._hf_major = transformers_major()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"   # required for batched generation

        kwargs: Dict[str, Any] = {"attn_implementation": "sdpa"}
        if load_in_4bit and device == "cuda":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            kwargs["device_map"] = {"": 0}
        else:
            # transformers >=5 renamed this argument
            dtype_key = "dtype" if self._hf_major >= 5 else "torch_dtype"
            kwargs[dtype_key] = torch_dtype if device == "cuda" else torch.float32
            kwargs["device_map"] = device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.last_stats: Dict[str, Any] = {}
        # Probed once, on the first generate(): see _generate_kwargs.
        self._logits_kwarg: Optional[str] = None
        self._probed = False

    def _chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_kwargs(self, enc) -> Dict[str, Any]:
        """Ask for ONE position's logits instead of the whole prefill.

        By default a causal LM materialises logits at every input position:
        2,473 tokens x 151,936 vocab x 4 bytes is ~1.5 GB per sequence, thrown
        away immediately because generation only needs the last position. That
        single tensor is what made batch 4 peak at 9 GB on a 6 GB card and
        spill over PCIe, which is why batching measured SLOWER than batch 1.

        The argument was renamed (`num_logits_to_keep` -> `logits_to_keep`), so
        probe once and remember which one this version accepts.
        """
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,                      # temperature 0
            "pad_token_id": self.tok.pad_token_id,
        }
        if self._logits_kwarg:
            kwargs[self._logits_kwarg] = 1
        return kwargs

    def _probe_logits_kwarg(self, enc) -> None:
        """One tiny generation per candidate name; keep the first that works."""
        torch = self.torch
        probe = {k: v[:1, -8:] for k, v in enc.items()}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            try:
                with torch.inference_mode():
                    self.model.generate(**probe, max_new_tokens=1, do_sample=False,
                                        pad_token_id=self.tok.pad_token_id, **{name: 1})
                self._logits_kwarg = name
                break
            except (TypeError, ValueError):
                continue
        self._probed = True

    def generate(self, prompts: Sequence[str], batch_size: Optional[int] = None) -> List[str]:
        torch = self.torch
        bs = batch_size or self.batch_size
        chats = [self._chat(p) for p in prompts]
        out: List[str] = []
        t0 = time.time()
        n_new = 0

        for start in range(0, len(chats), bs):
            batch = chats[start:start + bs]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            if not self._probed:
                self._probe_logits_kwarg(enc)
            with torch.inference_mode():
                generated = self.model.generate(**enc, **self._generate_kwargs(enc))
            new_tokens = generated[:, enc["input_ids"].shape[1]:]
            n_new += int(new_tokens.numel())
            out.extend(self.tok.batch_decode(new_tokens, skip_special_tokens=True))

        elapsed = time.time() - t0
        self.last_stats = {
            "n_prompts": len(prompts),
            "seconds": round(elapsed, 1),
            "prompts_per_second": round(len(prompts) / max(elapsed, 1e-9), 2),
            "new_tokens_per_second": round(n_new / max(elapsed, 1e-9), 1),
            "batch_size": bs,
            "logits_kwarg": self._logits_kwarg or "unsupported",
        }
        return out

    def describe(self) -> Dict[str, Any]:
        return {
            "generator": self.name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "greedy": True,
        }


class PatternStubGenerator(BaseGenerator):
    """Offline stand-in so the full pipeline runs with no model.

    It is a smoke-test device, not a baseline: it returns the first token in
    the context that looks like an identifier (letters followed by digits),
    which is exactly the shape of the planted answers in the synthetic corpus,
    and NO_ANSWER when there is none. That gives an offline run non-trivial EM
    and a non-trivial abstention rate, so the plumbing -- prompting, scoring,
    abstention accounting -- is genuinely exercised. Never report its numbers.
    """

    name = "pattern-stub"
    _TOKEN = re.compile(r"\b[A-Z][a-zA-Z]*\d{2,}\b")

    def __init__(self, no_answer: str = NO_ANSWER) -> None:
        self.no_answer = no_answer
        self.last_stats: Dict[str, Any] = {}

    def generate(self, prompts: Sequence[str], batch_size: Optional[int] = None) -> List[str]:
        t0 = time.time()
        out = []
        for prompt in prompts:
            context = prompt.split("Question:")[0]
            match = self._TOKEN.search(context)
            out.append(match.group(0) if match else self.no_answer)
        self.last_stats = {"n_prompts": len(prompts), "seconds": round(time.time() - t0, 3)}
        return out


def get_generator(
    model_name: str,
    system_prompt: str,
    offline: Optional[bool] = None,
    **kwargs: Any,
) -> BaseGenerator:
    if offline is None:
        offline = os.environ.get("CRAG_OFFLINE", "0") == "1"
    if offline:
        return PatternStubGenerator()
    return QwenGenerator(model_name, system_prompt, **kwargs)
