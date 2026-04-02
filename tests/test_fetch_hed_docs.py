"""Tests for the MyST markdown processing in fetch_hed_docs.py."""

import sys
from pathlib import Path

# Add scripts directory to path so we can import the fetch module
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from fetch_hed_docs import clean_myst_markdown


class TestCleanMystMarkdown:
    """Tests for clean_myst_markdown() directive stripping."""

    def test_passthrough_plain_markdown(self):
        """Plain markdown without directives passes through unchanged."""
        text = "# Heading\n\nSome paragraph.\n\n- List item\n"
        assert clean_myst_markdown(text) == text

    def test_empty_input(self):
        """Empty string returns empty string."""
        assert clean_myst_markdown("") == ""

    def test_strip_admonition_with_frontmatter(self):
        """Admonition with frontmatter is converted to bold title + body."""
        text = "```{admonition} My Title\n---\nclass: tip\n---\nBody content here.\n```\n"
        result = clean_myst_markdown(text)
        assert "**My Title**" in result
        assert "Body content here." in result
        assert "```{admonition}" not in result
        assert "class: tip" not in result

    def test_strip_admonition_without_frontmatter(self):
        """Admonition without frontmatter preserves title and body."""
        text = "```{admonition} Simple Note\nThis is the body.\n```\n"
        result = clean_myst_markdown(text)
        assert "**Simple Note**" in result
        assert "This is the body." in result

    def test_admonition_without_title(self):
        """Admonition with no title emits only body."""
        text = "```{admonition}\nJust a body.\n```\n"
        result = clean_myst_markdown(text)
        assert "Just a body." in result
        # No bold empty title
        assert "****" not in result

    def test_admonition_already_bold_title(self):
        """Admonition with bold markers in title avoids double-bolding."""
        text = "````{admonition} **Example:** A test case\nBody here.\n````\n"
        result = clean_myst_markdown(text)
        assert "**Example:** A test case" in result
        assert "****" not in result
        assert "Body here." in result

    def test_nested_code_in_admonition(self):
        """Code blocks inside admonitions are preserved."""
        text = (
            "````{admonition} Example\n"
            "---\n"
            "class: note\n"
            "---\n"
            "Some text.\n"
            "```\n"
            "code here\n"
            "```\n"
            "More text.\n"
            "````\n"
        )
        result = clean_myst_markdown(text)
        assert "code here" in result
        assert "Some text." in result
        assert "More text." in result

    def test_strip_list_table(self):
        """List-table is converted to pipe-delimited markdown table."""
        text = (
            "```{list-table}\n"
            "---\n"
            "header-rows: 1\n"
            "---\n"
            "* - Name\n"
            "  - Value\n"
            "* - Foo\n"
            "  - Bar\n"
            "```\n"
        )
        result = clean_myst_markdown(text)
        assert "| Name | Value |" in result
        assert "| Foo | Bar |" in result
        assert "| --- | --- |" in result
        assert "```{list-table}" not in result

    def test_list_table_with_multiline_cells(self):
        """List-table with continuation lines in cells."""
        text = (
            "```{list-table}\n"
            "---\n"
            "header-rows: 1\n"
            "---\n"
            "* - Tag\n"
            "  - Description\n"
            "* - `Definition`\n"
            "  - Used to name things\n"
            "    and create reusable patterns\n"
            "```\n"
        )
        result = clean_myst_markdown(text)
        assert "| `Definition` |" in result
        # Continuation lines should be joined
        assert "and create reusable patterns" in result

    def test_strip_code_block(self):
        """Code-block directive is converted to standard fenced code block."""
        text = "```{code-block} python\nx = 1\n```\n"
        result = clean_myst_markdown(text)
        assert "```python" in result
        assert "x = 1" in result
        assert "```{code-block}" not in result

    def test_code_block_without_language(self):
        """Code-block without language specifier."""
        text = "```{code-block}\nsome code\n```\n"
        result = clean_myst_markdown(text)
        assert "```" in result
        assert "some code" in result

    def test_strip_index_directive(self):
        """Index directives are stripped entirely."""
        text = "Before.\n```{index} Some term\nentry content\n```\nAfter.\n"
        result = clean_myst_markdown(text)
        assert "Before." in result
        assert "After." in result
        assert "Some term" not in result
        assert "entry content" not in result

    def test_strip_anchor_definitions(self):
        """Anchor definitions like (name)= are stripped."""
        text = "(my-anchor)=\n## My Section\nContent.\n"
        result = clean_myst_markdown(text)
        assert "## My Section" in result
        assert "Content." in result
        assert "(my-anchor)=" not in result

    def test_preserves_regular_parentheses(self):
        """Regular parentheses (not anchors) are preserved."""
        text = "This is (not an anchor) text.\n"
        result = clean_myst_markdown(text)
        assert "(not an anchor)" in result

    def test_four_backtick_fence(self):
        """Four-backtick fences are handled correctly."""
        text = "````{admonition} Test\nBody with ```code``` inside.\n````\n"
        result = clean_myst_markdown(text)
        assert "**Test**" in result
        assert "Body with ```code``` inside." in result

    def test_multiple_directives(self):
        """Multiple different directives in sequence."""
        text = (
            "# Heading\n"
            "\n"
            "```{admonition} Note\n"
            "First note.\n"
            "```\n"
            "\n"
            "Some text.\n"
            "\n"
            "```{admonition} Warning\n"
            "---\n"
            "class: warning\n"
            "---\n"
            "Second note.\n"
            "```\n"
        )
        result = clean_myst_markdown(text)
        assert "# Heading" in result
        assert "**Note**" in result
        assert "First note." in result
        assert "Some text." in result
        assert "**Warning**" in result
        assert "Second note." in result
        assert "class: warning" not in result
