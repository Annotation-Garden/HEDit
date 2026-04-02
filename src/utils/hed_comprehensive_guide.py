"""Comprehensive HED annotation guide for LLMs.

This module builds the system prompt for the annotation agent by combining
official HED documentation (fetched from hed-standard GitHub repos and
bundled with the package) with HEDit-specific sections for vocabulary
constraints, correction workflows, and output formatting.

The top-level get_comprehensive_hed_guide() assembles the full prompt.
Semantic hints are placed in the user prompt (not here) for prompt caching.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.hed_docs_loader import load_hed_docs

logger = logging.getLogger(__name__)


def format_semantic_hints(hints: list[dict]) -> str:
    """Format semantic hints for inclusion in the user prompt.

    Args:
        hints: List of semantic search results, each with:
              - tag: HED tag name
              - score: Relevance score (0-1)
              - source: Origin of the suggestion (e.g., "hed-lsp")
              - keyword: The input keyword that triggered this suggestion
              - prefix: Optional library prefix (e.g., "sc:"), unused in current workflow

    Returns:
        Formatted hints section for the user prompt
    """
    if not hints:
        return ""

    # Categorize by confidence level
    high_conf = []  # score >= 0.8
    medium_conf = []  # 0.5 <= score < 0.8
    low_conf = []  # score < 0.5

    for hint in hints:
        tag = hint.get("tag", "")
        if not tag:
            continue
        prefix = hint.get("prefix", "")
        score = hint.get("score", 0)
        full_tag = f"{prefix}{tag}" if prefix else tag

        if score >= 0.8:
            high_conf.append(full_tag)
        elif score >= 0.5:
            medium_conf.append(full_tag)
        else:
            low_conf.append(full_tag)

    lines = [
        "## SEMANTIC HINTS",
        "",
        "Based on your description, these schema tags may be relevant.",
        "Note: this list may contain false positives - use your judgment.",
        "",
    ]

    if high_conf:
        lines.append(f"**High confidence**: {', '.join(high_conf)}")
    if medium_conf:
        lines.append(f"**Medium confidence**: {', '.join(medium_conf)}")
    if low_conf:
        lines.append(f"**Lower confidence**: {', '.join(low_conf)}")

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def _build_no_extend_warning() -> str:
    """Build the no-extend warning section.

    Returns:
        Warning text for when extensions are prohibited
    """
    return """## EXTENSIONS STRICTLY PROHIBITED - USE ONLY EXISTING VOCABULARY

**ABSOLUTE RULE**: You MUST NOT create any new tags. Only use tags that exist in the vocabulary below.

THIS IS THE HIGHEST PRIORITY INSTRUCTION. IT OVERRIDES ALL EXAMPLES IN THIS GUIDE.

### What is FORBIDDEN:
- ANY tag with a slash (/) that creates a new concept (e.g., Animal/Marmoset, Animal/Dolphin, Vehicle/Rickshaw)
- Extending ANY parent tag with a new child term
- Creating new terms even if examples below suggest doing so

### What you MUST do instead:
- Use the MOST SIMILAR existing tag from vocabulary
- Use Label/description for clarification when needed
- Group with existing tags only

### EXAMPLES - NO EXTENSIONS MODE:

FORBIDDEN (extension): Animal/Marmoset, Animal/Dolphin, Building/Cottage
ALLOWED (existing tags): Animal, Animal-agent, Mammal (if in vocab)

FORBIDDEN: (Animal-agent, Animal/Marmoset)
ALLOWED: (Animal-agent, Animal) or (Animal-agent, Mammal) or (Animal-agent, Label/marmoset)

FORBIDDEN: Vehicle/Rickshaw
ALLOWED: Vehicle or (Vehicle, Label/rickshaw)

FORBIDDEN: Furniture/Armoire
ALLOWED: Furniture or (Furniture, Label/armoire)

The Label tag allows adding descriptive text without creating new schema tags.
Pattern: (Existing-tag, Label/description)

### Value tags with units ARE allowed:
Duration/2 s, Frequency/440 Hz - These are VALUES not extensions.

### Definitions ARE allowed (they don't create new schema tags):
Definition/MyDef, Def/MyDef - These are annotation tools, not extensions.

**REMINDER**: Ignore any examples below that show extensions like Animal/X or Building/Y.
Use only existing vocabulary tags. When in doubt, use Label/description.

---

"""


def _build_correction_workflow_section() -> str:
    """Build the correction workflow section for error-guided refinement.

    Returns:
        Correction workflow guidance text
    """
    return """## CORRECTION WORKFLOW

When fixing validation errors, follow these steps:

1. Read each error message carefully; it tells you exactly what is wrong and where.
2. For TAG_INVALID: the tag does not exist in the schema. Check the vocabulary list below
   and use the closest valid tag. If tag suggestions are provided, use those.
3. For TAG_EXTENSION_INVALID: you extended a tag that already exists as a vocabulary entry.
   Remove the parent prefix and use the tag directly (e.g., use "Press" not "Action/Press").
4. For TAG_REQUIRES_CHILD: the tag needs a value or child (e.g., "Duration/2 s" not just "Duration").
5. For PARENTHESES_MISMATCH: count opening and closing parentheses; each ( must have a matching ).
6. For VALUE_INVALID or UNITS_INVALID: use the correct value format with proper units.
7. If tag suggestions are provided, prefer those exact replacements over guessing.
8. Fix ALL reported errors in a single pass. Do not introduce new errors while fixing existing ones.
9. Preserve the semantic structure and meaning of the original annotation as much as possible.

---

"""


def _build_vocabulary_check_section() -> str:
    """Build the critical vocabulary check section.

    Returns:
        Vocabulary check instructions
    """
    return """## CRITICAL RULE: CHECK VOCABULARY FIRST

BEFORE using ANY tag with a slash (/), CHECK if it's in the vocabulary below!

WRONG: Item/Window, Item/Plant, Property/Red, Action/Press
RIGHT: Window, Plant, Red, Press (if these are in vocabulary)

The slash (/) is ONLY for:
1. NEW tags NOT in vocabulary: Building/Cottage (only if "Cottage" NOT in vocab)
2. Values with units: Duration/2 s, Frequency/440 Hz
3. Definitions: Definition/MyDef, Def/MyDef

IF YOU SEE TAG_EXTENSION_INVALID ERROR -> You extended a tag that exists in vocabulary!

---

"""


def _build_official_docs_section(docs: dict[str, str]) -> str:
    """Format bundled HED documentation for the system prompt.

    Inserts the official HedAnnotationSemantics.md and 02_Terminology.md
    content as labeled sections with source attribution.

    Args:
        docs: Dict from load_hed_docs() with keys
              "annotation_semantics" and "terminology"

    Returns:
        Formatted official docs section, or empty string if no docs available
    """
    sections = []

    if docs.get("annotation_semantics"):
        sections.append(
            "## OFFICIAL HED ANNOTATION SEMANTICS\n"
            "Source: https://www.hedtags.org/hed-resources/"
            "HedAnnotationSemantics.html\n\n"
            f"{docs['annotation_semantics']}\n\n---\n\n"
        )

    if docs.get("terminology"):
        sections.append(
            "## HED TERMINOLOGY\n"
            "Source: https://www.hedtags.org/hed-specification/"
            "02_Terminology.html\n\n"
            f"{docs['terminology']}\n\n---\n\n"
        )

    if not sections:
        logger.warning(
            "No official HED docs available for system prompt. "
            "Run 'python scripts/fetch_hed_docs.py' to fetch."
        )
        return ""

    return "".join(sections)


def _build_vocabulary_section(vocab_str: str, extend_str: str) -> str:
    """Build the vocabulary lookup section.

    Args:
        vocab_str: Comma-separated valid HED tags
        extend_str: Comma-separated extendable tags (or disabled message)

    Returns:
        Vocabulary section text
    """
    return f"""## VOCABULARY LOOKUP

ALWAYS check this list before using any tag. Use tags EXACTLY as shown.

{vocab_str}

CRITICAL:
- If "Press" is in this list -> use "Press" NOT "Action/Press"
- If "Button" is in this list -> use "Button" NOT "Item/Button"
- If "Circle" is in this list -> use "Circle" NOT "Item/Circle"
- If "Red" is in this list -> use "Red" NOT "Property/Red"

---

## EXTENDABLE TAGS

Only extend if the concept is NOT in vocabulary above.
When extending, use the MOST SPECIFIC applicable parent.

{extend_str}

---

"""


def _build_common_errors_section() -> str:
    """Build the common errors and troubleshooting section.

    Returns:
        Error troubleshooting text
    """
    return """## COMMON ERRORS AND TROUBLESHOOTING

### Error: TAG_EXTENSION_INVALID
CAUSE: Extending a tag with a child that already exists in schema vocabulary.

EXAMPLE ERRORS:
- Red-color/Red/DarkRed  (DarkRed may exist in vocab, use it directly)
- Sensory-presentation/Red  (Red exists in vocab, don't re-extend)
- Item/Window  (Window exists in vocab, use it directly)

FIX: Check vocabulary first. If tag exists, use it directly without slash extension.

WRONG: Building/House  (if House is in vocabulary)
RIGHT: House

WRONG: Action/Press  (if Press is in vocabulary)
RIGHT: Press

### Error: TAG_INVALID
CAUSE: Tag or extension is not valid in the schema.

EXAMPLE ERRORS:
- ReallyInvalid/Extension  (base tag doesn't exist)
- ReallyInvalid  (tag not in schema)
- Label #  (# used incorrectly outside sidecar)

FIX: Use only tags from the vocabulary or valid extensions from extendable tags.

WRONG: Stimulus/Visual  (Stimulus not in vocab)
RIGHT: Sensory-event, Visual-presentation

WRONG: Response/Button  (Response not a valid base)
RIGHT: Participant-response, (Press, Button)

### Error: VALUE_INVALID
CAUSE: Value substituted for placeholder (#) is incorrect format.

EXAMPLE ERRORS:
- Def/Acc/MyMy  (text instead of number for acceleration)
- Distance/4mxxx  (malformed unit)
- Duration/fast  (text instead of number)

FIX: Use correct value format with proper units.

WRONG: Duration/fast
RIGHT: Duration/2 s

WRONG: Frequency/high
RIGHT: Frequency/1000 Hz

WRONG: Distance/4mxxx
RIGHT: Distance/4 m

### Error: UNIT_CLASS_INVALID
CAUSE: Wrong unit type for the value.

EXAMPLE ERRORS:
- Duration/5 Hz  (Hz is frequency, not time)
- Frequency/3 s  (s is time, not frequency)

FIX: Match unit to tag's expected unit class.

Time units: s, ms, second, seconds, minute, minutes, hour
Frequency units: Hz, kHz, mHz
Distance units: m, cm, mm, km, ft, mile
Angle units: rad, deg, degree

### Error: CHARACTER_INVALID
CAUSE: Extension name contains invalid characters.

EXAMPLE ERRORS:
- Red/Red$2  ($ not allowed)
- Red/R#d  (# not allowed in extension names)

FIX: Use only letters, numbers, and hyphens in extension names.

WRONG: Animal/Cat$1
RIGHT: Animal/Cat-1 or Animal/Cat1

### Error: PARENTHESES_MISMATCH
CAUSE: Opening and closing parentheses don't match.

EXAMPLE ERRORS:
- ((Red, Circle)  (missing closing paren)
- (Red, Circle))  (extra closing paren)
- ((A, (B, C)))  (correct - properly nested)

FIX: Count parentheses; each ( must have matching ).

### Error: DEFINITION_INVALID
CAUSE: Definition used incorrectly.

EXAMPLE ERRORS:
- Definition/Name in HED column  (definitions only in sidecars)
- (Definition/X, (Def/Y))  (cannot nest Def inside Definition)
- (Definition/A, (Definition/B))  (cannot nest definitions)

FIX: Definitions only in sidecars, cannot contain Def or nested Definition.

### Quick Validation Checklist
Before submitting annotations:
1. Every tag exists in vocabulary OR is valid extension?
2. Extensions use most specific parent?
3. Event/Agent tags are NOT extended (use grouping)?
4. Value tags have proper units?
5. Parentheses are balanced?
6. Definitions only in sidecar, not event file?
7. Properties grouped with their objects?

---

"""


def _build_output_format_section() -> str:
    """Build the output format instructions section.

    Returns:
        Output format instructions text
    """
    return """## OUTPUT FORMAT

Output ONLY the HED annotation string.
Do NOT include:
- Markdown headers (##, ###)
- Code blocks (```)
- Explanatory text like "Here is", "Corrected", "Refined"
- Commentary or reasoning
- Line breaks within the annotation

Just output the raw HED annotation string directly.
"""


def get_comprehensive_hed_guide(
    vocabulary_sample: list[str],
    extendable_tags: list[str],
    no_extend: bool = False,
    *,
    docs_dir: Path | None = None,
) -> str:
    """Generate comprehensive HED annotation guide.

    Assembles HEDit-specific sections with official HED documentation
    into a complete system prompt for the annotation agent. The official
    docs (HedAnnotationSemantics.md and 02_Terminology.md) are bundled
    with the package and loaded at runtime.

    Note: Semantic hints are NOT included here to keep the system prompt
    static across requests, enabling prompt caching. Hints are passed
    in the user prompt instead.

    Args:
        vocabulary_sample: Full list of valid HED tags (complete vocabulary)
        extendable_tags: Tags that allow extension
        no_extend: If True, add strict instructions to prohibit tag extensions
        docs_dir: Optional override for docs directory (used by tests)

    Returns:
        Complete HED annotation guide
    """
    vocab_str = ", ".join(vocabulary_sample)
    extend_str = ", ".join(extendable_tags) if not no_extend else "(Extensions disabled)"

    # Load official HED documentation
    docs = load_hed_docs(docs_dir=docs_dir)

    # Format optional sections
    no_extend_warning = _build_no_extend_warning() if no_extend else ""

    # Assemble guide: HEDit-specific framing + official docs + vocabulary + errors
    # Note: semantic hints are placed in the user prompt for cache efficiency
    sections = [
        "# HED ANNOTATION GUIDE\n",
        no_extend_warning,
        _build_vocabulary_check_section(),
        _build_correction_workflow_section(),
        (
            "## SEMANTIC HINTS\n\n"
            "The user message may include a SEMANTIC HINTS section with "
            "potentially relevant tags from schema search. If present, use "
            "these as guidance for tag selection, but verify each against "
            "the vocabulary. If no hints section is present, proceed without them.\n\n"
            "---\n\n"
        ),
        _build_official_docs_section(docs),
        _build_vocabulary_section(vocab_str, extend_str),
        _build_common_errors_section(),
        _build_output_format_section(),
    ]

    return "".join(sections)
