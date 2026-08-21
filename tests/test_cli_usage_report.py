"""Tests for the CLI's token, cost, and cache-savings report."""

from src.cli import output
from src.cli.local_executor import LocalExecutionBackend
from src.utils.llm_usage import UsageLedger, summarize


def cached_usage() -> dict:
    """Usage summary for a run where most input came from the prompt cache."""
    ledger = UsageLedger()
    ledger.record(
        "annotation",
        "claude-haiku-4-5",
        {
            "input_tokens": 9000,
            "output_tokens": 200,
            "input_token_details": {"cache_read": 8000},
        },
    )
    ledger.record(
        "evaluation",
        "claude-haiku-4-5",
        {"input_tokens": 900, "output_tokens": 150},
    )
    return summarize(ledger)


def render(result: dict) -> str:
    """Render a result panel and return its text."""
    with output.console.capture() as capture:
        output.print_annotation_result(result, output_format="text")
    return capture.get()


class TestFormatUsageLines:
    """Tests for the usage line formatting."""

    def test_no_usage_reported(self):
        assert output.format_usage_lines(None) == []
        assert output.format_usage_lines({}) == []
        assert output.format_usage_lines({"calls": 0}) == []

    def test_reports_tokens_calls_and_cache(self):
        lines = output.format_usage_lines(cached_usage())

        assert "9,900 input / 350 output tokens in 2 LLM calls" in lines[0]
        assert "8,000 input tokens read from cache" in lines[1]
        assert "81% of input" in lines[1]

    def test_reports_savings_when_cache_was_read(self):
        cost_line = output.format_usage_lines(cached_usage())[2]

        assert "saved" in cost_line
        assert "by prompt caching" in cost_line

    def test_first_run_says_the_next_run_reuses_the_prompt(self):
        ledger = UsageLedger()
        ledger.record(
            "annotation",
            "claude-haiku-4-5",
            {"input_tokens": 9000, "output_tokens": 200, "input_token_details": {}},
        )
        lines = output.format_usage_lines(summarize(ledger))

        assert len(lines) == 2  # no cache line
        assert "nothing cached yet" in lines[1]

    def test_singular_call_wording(self):
        ledger = UsageLedger()
        ledger.record("annotation", "claude-haiku-4-5", {"input_tokens": 10, "output_tokens": 5})
        assert "1 LLM call" in output.format_usage_lines(summarize(ledger))[0]

    def test_unpriced_calls_are_flagged(self):
        ledger = UsageLedger()
        ledger.record("annotation", "claude-future-9", {"input_tokens": 500, "output_tokens": 20})
        cost_line = output.format_usage_lines(summarize(ledger))[-1]

        assert "excludes 1 call with no price" in cost_line

    def test_small_costs_are_not_rounded_to_zero(self):
        ledger = UsageLedger()
        ledger.record("keyword", "claude-haiku-4-5", {"input_tokens": 300, "output_tokens": 20})
        cost_line = output.format_usage_lines(summarize(ledger))[-1]

        assert cost_line.startswith("$0.000")
        assert cost_line != "$0.0000"


class TestAnnotationPanel:
    """Tests for the rendered result panel."""

    def test_panel_shows_the_savings_section(self):
        rendered = render(
            {
                "status": "success",
                "annotation": "Sensory-event, Visual-presentation",
                "is_valid": True,
                "is_faithful": True,
                "usage": cached_usage(),
            }
        )

        assert "Usage and cache savings" in rendered
        assert "read from cache" in rendered

    def test_panel_omits_the_section_without_usage(self):
        rendered = render(
            {
                "status": "success",
                "annotation": "Sensory-event",
                "is_valid": True,
                "is_faithful": True,
            }
        )

        assert "Usage and cache savings" not in rendered

    def test_status_markup_is_rendered_not_printed(self):
        """Rich markup in the status line must not leak into the output."""
        rendered = render(
            {
                "status": "failed",
                "annotation": "Sensory-event",
                "is_valid": False,
                "is_faithful": True,
                "is_complete": False,
                "validation_attempts": 2,
            }
        )

        assert "[green]" not in rendered
        assert "[/]" not in rendered
        # The ASCII checkboxes themselves survive escaping.
        assert "[x] Faithful" in rendered
        assert "[ ] Complete" in rendered

    def test_json_output_carries_usage_through(self, capsys):
        result = {"annotation": "Sensory-event", "usage": cached_usage()}
        output.print_annotation_result(result, output_format="json")

        printed = capsys.readouterr().out
        assert "cache_read_tokens" in printed
        assert "savings_usd" in printed


class TestStandaloneResultShape:
    """Tests for the result dictionary standalone mode returns."""

    @staticmethod
    def final_state() -> dict:
        return {
            "current_annotation": "Sensory-event, Visual-presentation",
            "is_valid": True,
            "is_faithful": True,
            "is_complete": True,
            "validation_attempts": 2,
            "validation_errors": [],
            "validation_warnings": ["Tag extension used"],
            "evaluation_feedback": "Faithful to the description",
            "assessment_feedback": "Complete",
            "total_iterations": 3,
        }

    def test_annotation_is_available_under_both_names(self):
        result = LocalExecutionBackend._shape_result(self.final_state(), "8.4.0", None)

        # hed_string is the backend contract; annotation is what the shared
        # text renderer reads.
        assert result["hed_string"] == "Sensory-event, Visual-presentation"
        assert result["annotation"] == result["hed_string"]

    def test_renderer_fields_are_populated(self):
        result = LocalExecutionBackend._shape_result(self.final_state(), "8.4.0", None)

        assert result["status"] == "success"
        assert result["is_faithful"] is True
        assert result["is_complete"] is True
        assert result["validation_attempts"] == 2
        assert result["validation_warnings"] == ["Tag extension used"]
        assert result["evaluation_feedback"] == "Faithful to the description"
        assert result["metadata"]["mode"] == "standalone"
        assert result["metadata"]["schema_version"] == "8.4.0"

    def test_validation_errors_appear_under_both_names(self):
        state = self.final_state()
        state["validation_errors"] = ["Unknown tag: Foo"]
        state["is_valid"] = False
        result = LocalExecutionBackend._shape_result(state, "8.4.0", None)

        assert result["status"] == "error"
        assert result["validation_errors"] == ["Unknown tag: Foo"]
        assert result["validation_messages"] == ["Unknown tag: Foo"]

    def test_usage_is_carried_and_renders(self):
        result = LocalExecutionBackend._shape_result(self.final_state(), "8.4.0", cached_usage())

        assert result["usage"]["cache_read_tokens"] == 8000
        assert "Usage and cache savings" in render(result)

    def test_extra_metadata_is_merged(self):
        result = LocalExecutionBackend._shape_result(
            self.final_state(),
            "8.4.0",
            None,
            extra_metadata={"vision_prompt": "Describe this image"},
        )

        assert result["metadata"]["vision_prompt"] == "Describe this image"
        assert result["metadata"]["mode"] == "standalone"
