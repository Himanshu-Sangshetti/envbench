"""Free local reference judge.

Runs on Apple Silicon via MLX with a quantised open-weight model, so the audit
needs no paid API and no network once the weights are cached.

Two uses in this project:
  1. **Reference judge** for the dynamic audit: the strong model against which
     each environment's own verifier is checked.
  2. **Rollout policy**: the weak-but-nonzero policy whose rollouts produce the
     Group B (learning-signal) statistics.

The model is loaded lazily and cached for the life of the process.

Usage:
    from envval.judge import LocalJudge
    j = LocalJudge()
    print(j.answer("Reply with exactly one word: yes or no. Is 2+2=4?"))
"""

from __future__ import annotations

import os
import re
import time
from typing import Sequence

DEFAULT_MODEL = os.environ.get("ENVVAL_JUDGE_MODEL", "mlx-community/Qwen3-8B-4bit")

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*", re.S)


class LocalJudge:
    """A local, free, strong-enough judge backed by MLX."""

    def __init__(self, model_id: str = DEFAULT_MODEL, max_tokens: int = 512):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None

    # -- model lifecycle -------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load

        t0 = time.time()
        self._model, self._tok = load(self.model_id)
        self.load_seconds = time.time() - t0

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # -- generation ------------------------------------------------------
    def _chat(self, messages: Sequence[dict], max_tokens: int | None = None) -> str:
        self.load()
        from mlx_lm import generate

        # Qwen3-style templates emit a <think> block before the answer, which
        # consumes the whole token budget on a short verdict. Disable thinking
        # where the template supports it.
        prompt = None
        for kw in ({"enable_thinking": False}, {}):
            try:
                prompt = self._tok.apply_chat_template(
                    list(messages), tokenize=False, add_generation_prompt=True, **kw
                )
                break
            except TypeError:
                continue
        if prompt is None:
            prompt = self._tok.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        raw = generate(
            self._model,
            self._tok,
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            verbose=False,
        )
        return _strip_think(raw)

    def answer(self, prompt: str, max_tokens: int | None = None) -> str:
        """Single-turn answer to a prompt."""
        return self._chat(
            [{"role": "user", "content": prompt}], max_tokens=max_tokens
        ).strip()

    def answer_batch(self, prompts: Sequence[str], max_tokens: int | None = None) -> list[str]:
        return [self.answer(p, max_tokens=max_tokens) for p in prompts]

    # -- judge protocols -------------------------------------------------
    def binary_judge(
        self,
        question: str,
        response: str,
        criteria: str = "The response is correct and responsive.",
    ) -> bool:
        """Strict yes/no judge. Returns True iff the model says yes.

        Deliberately conservative: anything that is not an explicit yes is a no,
        because an unanchored judge that defaults to yes is itself the defect
        this project measures.

        Empty or whitespace-only responses are rejected **without** consulting
        the model: a local judge was observed answering "yes" to an empty
        response, which is exactly the false-positive behaviour a reference
        judge must not have.
        """
        if not response or not response.strip():
            return False
        prompt = (
            "You are a strict grader. Answer with exactly one word: yes or no.\n\n"
            f"Question:\n{question}\n\n"
            f"Response:\n{response}\n\n"
            f"Criteria: {criteria}\n\n"
            "Is the response correct? Answer yes or no."
        )
        out = self.answer(prompt, max_tokens=8).lower()
        m = re.search(r"\b(yes|no)\b", out)
        return bool(m and m.group(1) == "yes")

    def rubric_judge(
        self,
        question: str,
        response: str,
        criteria: list[str],
    ) -> dict[str, bool]:
        """Per-criterion verdict, one criterion at a time (judge ensembles in
        the literature find single multi-criterion judges unreliable)."""
        out: dict[str, bool] = {}
        for c in criteria:
            out[c] = self.binary_judge(
                question, response, f"The response satisfies: {c}"
            )
        return out


def _strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _UNCLOSED_THINK_RE.sub("", text)
    return text.strip()


_shared: LocalJudge | None = None


def get_judge(model_id: str = DEFAULT_MODEL) -> LocalJudge:
    """Process-wide singleton so the model loads once."""
    global _shared
    if _shared is None or _shared.model_id != model_id:
        _shared = LocalJudge(model_id)
    return _shared
