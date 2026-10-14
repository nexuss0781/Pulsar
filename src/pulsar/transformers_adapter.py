"""Minimal Hugging Face Transformers adapter for Stage 6.

The adapter exposes the processed next-token categorical law, forkable GPT-style
KV states, batched one-token cache appends, and cached speculative-block target
rows. It is deliberately backend-specific: it targets decoder-only models that
accept legacy tuple ``past_key_values`` (GPT-2/DistilGPT-2 in Transformers 4.46).
The Stage 6 runner pins and reports that backend/version/model pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM
except ImportError as exc:  # pragma: no cover - exercised by the CLI dependency check
    raise ImportError(
        "Stage 6 requires torch and transformers; install the versions documented "
        "in algorithm_v6_backend.md"
    ) from exc


LegacyPast = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class KVState:
    """Model cache after consuming ``context`` and logits for its next token."""

    context: tuple[int, ...]
    past_key_values: LegacyPast
    next_logits: torch.Tensor


@dataclass
class BackendCounters:
    forward_calls: int = 0
    input_tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"forward_calls": int(self.forward_calls),
                "input_tokens": int(self.input_tokens)}


class TransformersCausalLM:
    """CPU/GPU decoder-only model with a narrow explicit cache API.

    The decoding law is softmax(logits / temperature), with no top-k/top-p,
    repetition penalty, or other logits processor. Consequently ``row`` is the
    exact target/draft categorical distribution used by the sampler.
    """

    def __init__(self, model_id: str, *, temperature: float = 1.0,
                 device: str = "cpu", max_batch_size: int = 32,
                 model: Any | None = None):
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.model_id = model_id
        self.temperature = float(temperature)
        self.device = torch.device(device)
        self.max_batch_size = int(max_batch_size)
        self.model = model or AutoModelForCausalLM.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.counters = BackendCounters()
        self.vocab_size = int(self.model.config.vocab_size)

    @staticmethod
    def _legacy(past: Any) -> LegacyPast:
        if hasattr(past, "to_legacy_cache"):
            past = past.to_legacy_cache()
        if not isinstance(past, (tuple, list)) or not past:
            raise TypeError("backend must return non-empty tuple-style KV cache")
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in past:
            if len(layer) < 2:
                raise TypeError("each KV layer must contain key and value tensors")
            layers.append((layer[0], layer[1]))
        return tuple(layers)

    @staticmethod
    def _clone_past(past: LegacyPast) -> LegacyPast:
        return tuple((key.clone(), value.clone()) for key, value in past)

    @staticmethod
    def _split_past(past: Any, batch_index: int) -> LegacyPast:
        legacy = TransformersCausalLM._legacy(past)
        return tuple((key[batch_index:batch_index + 1].contiguous(),
                      value[batch_index:batch_index + 1].contiguous())
                     for key, value in legacy)

    @staticmethod
    def _stack_pasts(states: Sequence[KVState]) -> LegacyPast:
        if not states:
            raise ValueError("cannot stack an empty KV state list")
        length = len(states[0].context)
        if any(len(state.context) != length for state in states):
            raise ValueError("batched cache states must have equal context lengths")
        return tuple((torch.cat([state.past_key_values[layer][0]
                                 for state in states], dim=0),
                      torch.cat([state.past_key_values[layer][1]
                                 for state in states], dim=0))
                     for layer in range(len(states[0].past_key_values)))

    @staticmethod
    def _crop_past(past: Any, length: int) -> LegacyPast:
        legacy = TransformersCausalLM._legacy(past)
        if length <= 0:
            raise ValueError("KV cache length must be positive")
        cropped = []
        for key, value in legacy:
            if key.shape[-2] < length or value.shape[-2] < length:
                raise ValueError("cannot crop a cache shorter than the requested prefix")
            cropped.append((key[..., :length, :].contiguous(),
                            value[..., :length, :].contiguous()))
        return tuple(cropped)

    def _forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                 past_key_values: LegacyPast | None = None,
                 position_ids: torch.Tensor | None = None):
        self.counters.forward_calls += 1
        self.counters.input_tokens += int(input_ids.numel())
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                position_ids=position_ids,
                                past_key_values=past_key_values,
                                use_cache=True, return_dict=True)
        if output.past_key_values is None:
            raise RuntimeError("model did not return past_key_values")
        return output

    def row(self, logits: torch.Tensor) -> np.ndarray:
        """Return the normalized processed categorical law for one logit row."""
        if logits.ndim != 1 or int(logits.numel()) != self.vocab_size:
            raise ValueError("logit row has the wrong rank or vocabulary size")
        values = logits.detach().to(dtype=torch.float64, device="cpu").numpy()
        values = values / self.temperature
        values -= np.max(values)
        weights = np.exp(values)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("model logits did not produce a finite positive law")
        return weights / total

    def prefill(self, context: Sequence[int]) -> KVState:
        ids = tuple(int(token) for token in context)
        if not ids:
            raise ValueError("context must contain at least one token")
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(input_ids)
        output = self._forward(input_ids=input_ids, attention_mask=mask)
        return KVState(ids, self._split_past(output.past_key_values, 0),
                       output.logits[0, -1].detach())

    def prefill_many(self, contexts: Sequence[Sequence[int]]) -> list[KVState]:
        """Prefill equal-length full contexts in bounded batches."""
        if not contexts:
            return []
        ids = [tuple(int(token) for token in context) for context in contexts]
        if any(not context for context in ids):
            raise ValueError("contexts must be non-empty")
        lengths = {len(context) for context in ids}
        if len(lengths) != 1:
            raise ValueError("prefill_many requires equal-length contexts")
        results: list[KVState | None] = [None] * len(ids)
        for begin in range(0, len(ids), self.max_batch_size):
            end = min(len(ids), begin + self.max_batch_size)
            batch_ids = torch.tensor(ids[begin:end], dtype=torch.long, device=self.device)
            mask = torch.ones_like(batch_ids)
            output = self._forward(input_ids=batch_ids, attention_mask=mask)
            for local, absolute in enumerate(range(begin, end)):
                results[absolute] = KVState(
                    ids[absolute], self._split_past(output.past_key_values, local),
                    output.logits[local, -1].detach())
        return [state for state in results if state is not None]

    def fork(self, state: KVState) -> KVState:
        return KVState(state.context, self._clone_past(state.past_key_values),
                       state.next_logits.clone())

    def append(self, state: KVState, token: int) -> KVState:
        """Consume one committed token and return a new immutable-style state."""
        token = int(token)
        length = len(state.context)
        input_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
        mask = torch.ones((1, length + 1), dtype=torch.long, device=self.device)
        positions = torch.tensor([[length]], dtype=torch.long, device=self.device)
        output = self._forward(input_ids=input_ids, attention_mask=mask,
                               past_key_values=state.past_key_values,
                               position_ids=positions)
        return KVState(state.context + (token,), self._legacy(output.past_key_values),
                       output.logits[0, -1].detach())

    def append_many(self, states: Sequence[KVState], tokens: Sequence[int]) -> list[KVState]:
        """Batched branch expansion: append one token to each equal-length cache."""
        if len(states) != len(tokens):
            raise ValueError("states and tokens must have equal lengths")
        if not states:
            return []
        length = len(states[0].context)
        if any(len(state.context) != length for state in states):
            raise ValueError("batched appends require equal-length contexts")
        token_ids = [int(token) for token in tokens]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        mask = torch.ones((len(states), length + 1), dtype=torch.long, device=self.device)
        positions = torch.full((len(states), 1), length, dtype=torch.long,
                               device=self.device)
        output = self._forward(input_ids=input_ids, attention_mask=mask,
                               past_key_values=self._stack_pasts(states),
                               position_ids=positions)
        legacy = self._legacy(output.past_key_values)
        result = []
        for i, (state, token) in enumerate(zip(states, token_ids)):
            result.append(KVState(
                state.context + (token,), self._split_past(legacy, i),
                output.logits[i, -1].detach()))
        return result

    def verify_chain(self, root: KVState, proposals: Sequence[int]
                     ) -> tuple[list[np.ndarray], list[KVState]]:
        """Get p rows and prefix KV states for a speculative proposal block.

        The root distribution is already in ``root``. A single target call
        consumes the proposal sequence; its intermediate logits are exactly the
        target rows after each prefix, and the returned cache is cropped to each
        committed prefix if verification rejects early.
        """
        proposals = tuple(int(token) for token in proposals)
        rows = [self.row(root.next_logits)]
        states = [root]
        if not proposals:
            return rows, states
        length = len(root.context)
        input_ids = torch.tensor([proposals], dtype=torch.long, device=self.device)
        mask = torch.ones((1, length + len(proposals)), dtype=torch.long,
                          device=self.device)
        positions = torch.arange(length, length + len(proposals), dtype=torch.long,
                                 device=self.device).unsqueeze(0)
        output = self._forward(input_ids=input_ids, attention_mask=mask,
                               past_key_values=root.past_key_values,
                               position_ids=positions)
        for i in range(1, len(proposals) + 1):
            prefix = proposals[:i]
            logits = output.logits[0, i - 1].detach()
            cache = self._crop_past(output.past_key_values, length + i)
            states.append(KVState(root.context + prefix, cache, logits))
            rows.append(self.row(logits))
        return rows, states

    def reset_counters(self) -> None:
        self.counters = BackendCounters()

    def counter_snapshot(self) -> dict[str, int]:
        return self.counters.as_dict()


def assert_compatible_vocab(target: TransformersCausalLM,
                            draft: TransformersCausalLM, tokenizer_size: int) -> None:
    if target.vocab_size != draft.vocab_size or target.vocab_size != tokenizer_size:
        raise ValueError(
            f"target/draft/tokenizer vocabularies differ: {target.vocab_size}, "
            f"{draft.vocab_size}, {tokenizer_size}"
        )


def top_positive(row: np.ndarray, count: int) -> list[int]:
    """Descending probability/token-ID order, matching the Stage 3 tie contract."""
    if count < 0:
        raise ValueError("branch count must be non-negative")
    if count == 0:
        return []
    count = min(count, len(row))
    candidates = np.argpartition(-row, count - 1)[:count]
    candidates = candidates[row[candidates] > 0.0]
    order = np.lexsort((candidates, -row[candidates]))
    return [int(token) for token in candidates[order]]


def verified_sample(target_row: np.ndarray, proposal_row: np.ndarray,
                    rng: np.random.Generator) -> tuple[int, int, bool]:
    """One exact v1 rejection step; returns (proposal, emitted, accepted)."""
    p = np.asarray(target_row, dtype=np.float64)
    q = np.asarray(proposal_row, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or p.size == 0:
        raise ValueError("target and proposal rows must be non-empty matching vectors")
    if (np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)) or np.any(p < 0) or
            np.any(q < 0) or p.sum() <= 0 or q.sum() <= 0):
        raise ValueError("target and proposal rows must be finite positive laws")
    p = p / p.sum()
    q = q / q.sum()
    proposal = int(rng.choice(len(q), p=q))
    return verify_proposal(p, q, proposal, rng, normalized=True)


def verify_proposal(target_row: np.ndarray, proposal_row: np.ndarray, proposal: int,
                    rng: np.random.Generator, *, normalized: bool = False
                    ) -> tuple[int, int, bool]:
    """Verify a proposal drawn earlier from q, as in speculative-chain decoding."""
    p = np.asarray(target_row, dtype=np.float64)
    q = np.asarray(proposal_row, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or p.size == 0:
        raise ValueError("target and proposal rows must be non-empty matching vectors")
    if (np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)) or np.any(p < 0) or
            np.any(q < 0) or p.sum() <= 0 or q.sum() <= 0):
        raise ValueError("target and proposal rows must be finite positive laws")
    if not normalized:
        p = p / p.sum()
        q = q / q.sum()
    proposal = int(proposal)
    if proposal < 0 or proposal >= len(q) or q[proposal] <= 0.0:
        raise ValueError("the fixed proposal must have positive q probability")
    acceptance = min(1.0, float(p[proposal] / q[proposal]))
    if float(rng.random()) < acceptance:
        return proposal, proposal, True
    residual = np.maximum(p - q, 0.0)
    mass = float(residual.sum())
    if mass <= 0.0:
        raise ArithmeticError("rejection occurred with an empty residual")
    emitted = int(rng.choice(len(residual), p=residual / mass))
    return proposal, emitted, False


def exactness_identity_error(target_row: np.ndarray,
                             proposal_row: np.ndarray) -> float:
    """Maximum pointwise |min(p,q) + (p-q)+ - p| after normalization."""
    p = np.asarray(target_row, dtype=np.float64)
    q = np.asarray(proposal_row, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or p.size == 0:
        raise ValueError("target and proposal rows must match")
    p = p / p.sum()
    q = q / q.sum()
    reconstructed = np.minimum(p, q) + np.maximum(p - q, 0.0)
    return float(np.max(np.abs(reconstructed - p)))
