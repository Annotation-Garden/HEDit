"""Tests for token and prompt-cache accounting."""

import asyncio

import pytest

from src.utils.llm_usage import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    PRICING_USD_PER_MTOK,
    UsageLedger,
    UsageTotals,
    caching_expected,
    process_ledger,
    record_usage,
    split_input_tokens,
    usage_scope,
)


def usage_metadata(
    input_tokens: int,
    output_tokens: int = 0,
    cache_read: int | None = None,
    cache_creation: int | None = None,
    ephemeral_5m: int | None = None,
) -> dict:
    """Build a LangChain usage-metadata mapping.

    ``input_tokens`` is the true total (LangChain adds cached tokens back
    into Anthropic's cache-excluding count), matching what
    langchain-anthropic emits.
    """
    details = {}
    if cache_read is not None:
        details["cache_read"] = cache_read
    if cache_creation is not None:
        details["cache_creation"] = cache_creation
    if ephemeral_5m is not None:
        details["ephemeral_5m_input_tokens"] = ephemeral_5m

    usage: dict = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if details:
        usage["input_token_details"] = details
    return usage


class TestSplitInputTokens:
    """Tests for splitting LangChain usage metadata into billing buckets."""

    def test_generic_cache_creation_key(self):
        usage = usage_metadata(5000, cache_read=3000, cache_creation=1500)
        assert split_input_tokens(usage) == (500, 3000, 1500)

    def test_per_ttl_buckets_are_summed(self):
        # langchain-anthropic zeroes the generic key when Anthropic reports
        # per-TTL buckets, so both keys must be added.
        usage = usage_metadata(5000, cache_read=1000, cache_creation=0, ephemeral_5m=3500)
        assert split_input_tokens(usage) == (500, 1000, 3500)

    def test_no_cache_details(self):
        assert split_input_tokens(usage_metadata(1200)) == (1200, 0, 0)

    def test_uncached_never_negative(self):
        # Defensive: a provider reporting more cached than total input must
        # not produce a negative uncached count.
        usage = usage_metadata(1000, cache_read=1200)
        assert split_input_tokens(usage) == (0, 1200, 0)


class TestCostAccounting:
    """Tests for the cost and savings arithmetic."""

    def test_cache_read_billed_at_one_tenth(self):
        ledger = UsageLedger()
        ledger.record(
            "annotation",
            "claude-haiku-4-5",
            usage_metadata(5000, output_tokens=500, cache_read=4000),
        )
        totals = ledger.total()

        input_price, output_price = PRICING_USD_PER_MTOK["claude-haiku-4-5"]
        expected_cost = (
            1000 + 4000 * CACHE_READ_MULTIPLIER
        ) * input_price / 1_000_000 + 500 * output_price / 1_000_000
        expected_uncached = 5000 * input_price / 1_000_000 + 500 * output_price / 1_000_000

        assert totals.cost_usd == pytest.approx(expected_cost)
        assert totals.uncached_cost_usd == pytest.approx(expected_uncached)
        assert totals.savings_usd == pytest.approx(expected_uncached - expected_cost)
        assert totals.cache_hit_rate == pytest.approx(0.8)

    def test_cache_write_carries_a_premium(self):
        ledger = UsageLedger()
        ledger.record(
            "annotation",
            "claude-haiku-4-5",
            usage_metadata(5000, cache_creation=4096),
        )
        totals = ledger.total()

        input_price, _ = PRICING_USD_PER_MTOK["claude-haiku-4-5"]
        billed = 904 + 4096 * CACHE_WRITE_MULTIPLIER
        assert totals.cost_usd == pytest.approx(billed * input_price / 1_000_000)
        # A write costs more than the uncached price, so the first request of
        # a session shows negative raw savings, clamped to zero.
        assert totals.cost_usd > totals.uncached_cost_usd
        assert totals.savings_usd == 0.0
        assert totals.savings_pct == 0.0

    def test_no_caching_means_no_savings(self):
        ledger = UsageLedger()
        ledger.record("keyword", "claude-haiku-4-5", usage_metadata(800, output_tokens=40))
        totals = ledger.total()

        assert totals.cache_read_tokens == 0
        assert totals.cost_usd == pytest.approx(totals.uncached_cost_usd)
        assert totals.savings_usd == 0.0

    def test_sonnet_priced_higher_than_haiku(self):
        haiku = UsageLedger()
        haiku.record("annotation", "claude-haiku-4-5", usage_metadata(1000, output_tokens=100))
        sonnet = UsageLedger()
        sonnet.record("annotation", "claude-sonnet-5", usage_metadata(1000, output_tokens=100))

        assert sonnet.total().cost_usd > haiku.total().cost_usd

    def test_unpriced_model_keeps_tokens_but_not_cost(self):
        ledger = UsageLedger()
        ledger.record("annotation", "claude-experimental-9", usage_metadata(1000, output_tokens=50))
        totals = ledger.total()

        assert totals.input_tokens == 1000
        assert totals.output_tokens == 50
        assert totals.cost_usd == 0.0
        assert totals.unpriced_calls == 1


class TestUsageTotals:
    """Tests for the totals value object."""

    def test_input_tokens_include_cached(self):
        totals = UsageTotals(
            uncached_input_tokens=100,
            cache_read_tokens=4000,
            cache_write_tokens=200,
            output_tokens=50,
        )
        assert totals.input_tokens == 4300
        assert totals.total_tokens == 4350

    def test_empty_totals_have_no_rates(self):
        totals = UsageTotals()
        assert totals.cache_hit_rate == 0.0
        assert totals.savings_pct == 0.0
        assert totals.savings_usd == 0.0

    def test_add_merges_models(self):
        first = UsageTotals(calls=1, output_tokens=10, models={"claude-haiku-4-5"})
        second = UsageTotals(calls=2, output_tokens=5, models={"claude-sonnet-5"})
        first.add(second)

        assert first.calls == 3
        assert first.output_tokens == 15
        assert first.models == {"claude-haiku-4-5", "claude-sonnet-5"}

    def test_as_dict_is_json_safe(self):
        totals = UsageTotals(calls=1, cache_read_tokens=4000, models={"claude-haiku-4-5"})
        payload = totals.as_dict()

        assert payload["cache_read_tokens"] == 4000
        assert payload["models"] == ["claude-haiku-4-5"]
        assert isinstance(payload["cache_hit_rate"], float)


class TestLedgerBreakdown:
    """Tests for per-role and per-model grouping."""

    def test_roles_and_models_tracked_separately(self):
        ledger = UsageLedger()
        ledger.record("annotation", "claude-sonnet-5", usage_metadata(6000, cache_read=5000))
        ledger.record("evaluation", "claude-haiku-4-5", usage_metadata(900, output_tokens=120))
        ledger.record("evaluation", "claude-haiku-4-5", usage_metadata(900, output_tokens=100))

        by_role = ledger.by_role()
        assert by_role["annotation"].calls == 1
        assert by_role["evaluation"].calls == 2
        assert by_role["evaluation"].output_tokens == 220

        by_model = ledger.by_model()
        assert set(by_model) == {"claude-sonnet-5", "claude-haiku-4-5"}
        assert ledger.total().calls == 3

    def test_snapshot_shape(self):
        ledger = UsageLedger()
        ledger.record("annotation", "claude-haiku-4-5", usage_metadata(5000, cache_read=4096))
        snapshot = ledger.snapshot()

        assert set(snapshot) == {"total", "by_role", "by_model"}
        assert snapshot["by_role"]["annotation"]["cache_read_tokens"] == 4096
        assert snapshot["total"]["calls"] == 1

    def test_empty_and_reset(self):
        ledger = UsageLedger()
        assert ledger.is_empty()
        ledger.record("annotation", "claude-haiku-4-5", usage_metadata(10))
        assert not ledger.is_empty()
        ledger.reset()
        assert ledger.is_empty()
        assert ledger.total().calls == 0


class TestUsageScope:
    """Tests for per-request usage collection."""

    def test_scope_sees_only_its_own_calls(self):
        process_ledger().reset()

        with usage_scope() as first:
            record_usage("annotation", "claude-haiku-4-5", usage_metadata(1000))
        with usage_scope() as second:
            record_usage("evaluation", "claude-haiku-4-5", usage_metadata(500))

        assert first.total().input_tokens == 1000
        assert second.total().input_tokens == 500
        # The process ledger accumulates across scopes.
        assert process_ledger().total().input_tokens == 1500

    def test_calls_outside_a_scope_still_reach_the_process_ledger(self):
        process_ledger().reset()
        record_usage("vision", "claude-haiku-4-5", usage_metadata(300))

        assert process_ledger().total().calls == 1

    def test_nested_scopes_are_independent(self):
        with usage_scope() as outer:
            record_usage("annotation", "claude-haiku-4-5", usage_metadata(100))
            with usage_scope() as inner:
                record_usage("evaluation", "claude-haiku-4-5", usage_metadata(50))
            record_usage("assessment", "claude-haiku-4-5", usage_metadata(25))

        assert inner.total().input_tokens == 50
        # The inner block's calls belong to the inner scope only.
        assert outer.total().input_tokens == 125
        assert set(outer.by_role()) == {"annotation", "assessment"}

    def test_scope_is_restored_after_an_exception(self):
        with pytest.raises(RuntimeError):
            with usage_scope():
                raise RuntimeError("workflow failed")

        # No active scope leaks into later calls.
        process_ledger().reset()
        record_usage("annotation", "claude-haiku-4-5", usage_metadata(10))
        assert process_ledger().total().calls == 1

    def test_scope_covers_concurrent_tasks(self):
        """Workflow nodes run as separate tasks; their usage must be counted."""

        async def call(role: str, tokens: int) -> None:
            await asyncio.sleep(0)
            record_usage(role, "claude-haiku-4-5", usage_metadata(tokens))

        async def run() -> UsageLedger:
            with usage_scope() as ledger:
                await asyncio.gather(
                    call("annotation", 400),
                    call("keyword", 100),
                    call("evaluation", 250),
                )
            return ledger

        ledger = asyncio.run(run())
        assert ledger.total().calls == 3
        assert ledger.total().input_tokens == 750


class TestCachingExpected:
    """Tests for the minimum cacheable prefix check."""

    def test_haiku_needs_4096_tokens(self):
        assert not caching_expected("claude-haiku-4-5", 4095)
        assert caching_expected("claude-haiku-4-5", 4096)

    def test_sonnet_needs_1024_tokens(self):
        assert not caching_expected("claude-sonnet-5", 1023)
        assert caching_expected("claude-sonnet-5", 1024)

    def test_unknown_model_is_not_claimed_uncacheable(self):
        assert caching_expected("claude-future-9", 10)
