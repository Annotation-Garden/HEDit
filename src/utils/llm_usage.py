"""Token and prompt-cache accounting for HEDit's LLM calls.

Every LLM in HEDit is built by :func:`src.utils.anthropic_llm.create_anthropic_llm`
and wrapped in ``CachingLLMWrapper``, which reports each response's usage
metadata here. Two views are kept:

- **Process totals** since startup, broken down by agent role and by model.
  Served by ``GET /metrics``.
- **Per-request totals**, for code running inside :func:`usage_scope`. The
  CLI efficiency report and annotation telemetry both read their numbers
  from a request scope.

Why the cache counters matter: on the Messages API a cache read costs 0.1x
the base input price and a 5-minute cache write costs 1.25x, so a workflow
that re-sends the same large HED vocabulary guide on every call pays a
fraction of the list price. A prefix shorter than the model's minimum
(4096 tokens on Haiku 4.5, 1024 on Sonnet 5) silently does not cache at
all: ``cache_creation`` stays 0 and those tokens bill as ordinary input,
with no error. Reading these counters is the only reliable way to know
which of the two happened.
"""

import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Cache-read tokens bill at 0.1x the base input price; a 5-minute cache
# write costs 1.25x (a 1-hour write, which HEDit does not use, costs 2x).
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25

# Minimum cacheable prefix per model. A shorter system prompt is sent with
# its cache_control marker intact but never creates a cache entry.
MIN_CACHEABLE_PREFIX_TOKENS = {
    "claude-haiku-4-5": 4096,
    "claude-sonnet-5": 1024,
}

# USD per million tokens, (input, output). Anthropic list prices; Sonnet 5
# has promotional pricing ($2/$10) through 2026-08-31, so using list prices
# means reported cost is an upper bound rather than an understatement.
PRICING_USD_PER_MTOK = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}


@dataclass
class UsageTotals:
    """Token counts and cost for a set of LLM calls.

    ``uncached_cost_usd`` is what the same calls would have cost with no
    prompt caching at all (every input token at full price), which is what
    makes the savings figure meaningful rather than self-congratulatory.
    """

    calls: int = 0
    uncached_input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    uncached_cost_usd: float = 0.0
    unpriced_calls: int = 0
    models: set[str] = field(default_factory=set)

    @property
    def input_tokens(self) -> int:
        """Total input tokens, including tokens served from cache."""
        return self.uncached_input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Share of input tokens served from cache (0.0-1.0)."""
        if self.input_tokens == 0:
            return 0.0
        return self.cache_read_tokens / self.input_tokens

    @property
    def savings_usd(self) -> float:
        """Dollars not spent because of prompt caching."""
        return max(self.uncached_cost_usd - self.cost_usd, 0.0)

    @property
    def savings_pct(self) -> float:
        """Savings as a share of the uncached cost (0.0-1.0)."""
        if self.uncached_cost_usd <= 0:
            return 0.0
        return self.savings_usd / self.uncached_cost_usd

    def add(self, other: "UsageTotals") -> None:
        """Accumulate another set of totals into this one."""
        self.calls += other.calls
        self.uncached_input_tokens += other.uncached_input_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        self.uncached_cost_usd += other.uncached_cost_usd
        self.unpriced_calls += other.unpriced_calls
        self.models |= other.models

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable view, safe to embed in API responses."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "uncached_input_tokens": self.uncached_input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "cost_usd": round(self.cost_usd, 6),
            "uncached_cost_usd": round(self.uncached_cost_usd, 6),
            "savings_usd": round(self.savings_usd, 6),
            "savings_pct": round(self.savings_pct, 4),
            "models": sorted(self.models),
            "unpriced_calls": self.unpriced_calls,
        }


def split_input_tokens(usage: Mapping[str, Any]) -> tuple[int, int, int]:
    """Split LangChain usage metadata into (uncached, cache_read, cache_write).

    LangChain reports ``input_tokens`` as the true total (Anthropic's own
    ``input_tokens`` excludes cached tokens, so langchain-anthropic adds
    them back). Cache writes arrive either under the generic
    ``cache_creation`` key or, when Anthropic reports per-TTL buckets, under
    the ``ephemeral_*`` keys with the generic key zeroed to avoid double
    counting; summing all three is correct in both shapes.

    Args:
        usage: A LangChain ``UsageMetadata`` mapping

    Returns:
        Tuple of (uncached input tokens, cache-read tokens, cache-write tokens)
    """
    details = usage.get("input_token_details") or {}
    cache_read = int(details.get("cache_read") or 0)
    cache_write = sum(
        int(details.get(key) or 0)
        for key in ("cache_creation", "ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
    )
    total_input = int(usage.get("input_tokens") or 0)
    uncached = max(total_input - cache_read - cache_write, 0)
    return uncached, cache_read, cache_write


def _totals_for_call(model: str, usage: Mapping[str, Any]) -> UsageTotals:
    """Build the totals contributed by a single LLM response."""
    uncached, cache_read, cache_write = split_input_tokens(usage)
    output = int(usage.get("output_tokens") or 0)

    totals = UsageTotals(
        calls=1,
        uncached_input_tokens=uncached,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        output_tokens=output,
        models={model} if model else set(),
    )

    pricing = PRICING_USD_PER_MTOK.get(model)
    if pricing is None:
        # An unpriced model still contributes token counts; leaving cost at
        # zero and counting the call keeps the dollar figures honest.
        totals.unpriced_calls = 1
        return totals

    input_price, output_price = pricing
    billed_input = (
        uncached + cache_read * CACHE_READ_MULTIPLIER + cache_write * CACHE_WRITE_MULTIPLIER
    )
    output_cost = output * output_price / 1_000_000
    totals.cost_usd = billed_input * input_price / 1_000_000 + output_cost
    totals.uncached_cost_usd = (
        uncached + cache_read + cache_write
    ) * input_price / 1_000_000 + output_cost
    return totals


class UsageLedger:
    """Accumulates LLM usage by agent role and by model.

    Instances are used two ways: one process-wide ledger for server metrics,
    and short-lived per-request ledgers created by :func:`usage_scope`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_role: dict[str, UsageTotals] = {}
        self._by_model: dict[str, UsageTotals] = {}

    def record(self, role: str, model: str, usage: Mapping[str, Any]) -> UsageTotals:
        """Record one LLM response.

        Args:
            role: Agent role that made the call (e.g. "annotation")
            model: Model id that served the call
            usage: LangChain ``UsageMetadata`` mapping

        Returns:
            The totals contributed by this single call
        """
        call = _totals_for_call(model, usage)
        with self._lock:
            self._by_role.setdefault(role, UsageTotals()).add(call)
            if model:
                self._by_model.setdefault(model, UsageTotals()).add(call)
        return call

    def total(self) -> UsageTotals:
        """Combined totals across every role."""
        combined = UsageTotals()
        with self._lock:
            for totals in self._by_role.values():
                combined.add(totals)
        return combined

    def by_role(self) -> dict[str, UsageTotals]:
        with self._lock:
            return dict(self._by_role)

    def by_model(self) -> dict[str, UsageTotals]:
        with self._lock:
            return dict(self._by_model)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._by_role

    def snapshot(self) -> dict[str, Any]:
        """JSON-serializable view of totals, roles, and models."""
        return {
            "total": self.total().as_dict(),
            "by_role": {role: totals.as_dict() for role, totals in self.by_role().items()},
            "by_model": {model: totals.as_dict() for model, totals in self.by_model().items()},
        }

    def reset(self) -> None:
        """Drop all recorded usage (used by tests)."""
        with self._lock:
            self._by_role.clear()
            self._by_model.clear()


_process_ledger = UsageLedger()
_active_scope: ContextVar[UsageLedger | None] = ContextVar("hedit_usage_scope", default=None)


def process_ledger() -> UsageLedger:
    """The process-wide ledger, covering every call since startup."""
    return _process_ledger


def record_usage(role: str, model: str, usage: Mapping[str, Any]) -> None:
    """Record one LLM response in the process ledger and the active scope."""
    call = _process_ledger.record(role, model, usage)

    scope = _active_scope.get()
    if scope is not None:
        scope.record(role, model, usage)

    logger.debug(
        "%s call on %s: input=%d (cache_read=%d, cache_write=%d), output=%d, cost=$%.6f",
        role,
        model or "unknown",
        call.input_tokens,
        call.cache_read_tokens,
        call.cache_write_tokens,
        call.output_tokens,
        call.cost_usd,
    )


@contextmanager
def usage_scope() -> Iterator[UsageLedger]:
    """Collect usage for the calls made inside this block.

    The scope is stored in a context variable, so it covers awaited
    coroutines and tasks spawned within the block (LangGraph nodes included)
    without threading a ledger through the workflow. Nested scopes are
    independent: the inner scope sees only its own calls, while the process
    ledger sees everything.
    """
    ledger = UsageLedger()
    token = _active_scope.set(ledger)
    try:
        yield ledger
    finally:
        _active_scope.reset(token)


def caching_expected(model: str, prefix_tokens: int) -> bool:
    """Whether a prefix of this size can create a cache entry on this model.

    Args:
        model: First-party Anthropic model id
        prefix_tokens: Size of the cached prefix in tokens

    Returns:
        True when the prefix clears the model's minimum cacheable size.
        Unknown models return True (no basis to claim otherwise).
    """
    minimum = MIN_CACHEABLE_PREFIX_TOKENS.get(model)
    if minimum is None:
        return True
    return prefix_tokens >= minimum
