#!/usr/bin/env python3
"""Fetch and process official HED documentation from GitHub.

Downloads HedAnnotationSemantics.md and 02_Terminology.md from the
hed-standard GitHub repositories, processes them to strip MyST/Sphinx
directives, and saves them as clean markdown in src/data/hed-docs/.

Usage:
    python scripts/fetch_hed_docs.py [--force]

The script is idempotent: it skips writing if the content hash is unchanged.
Use --force to overwrite regardless.
"""

import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx is required: uv pip install httpx", file=sys.stderr)
    sys.exit(1)

# Official HED documentation sources
DOCS = [
    {
        "filename": "HedAnnotationSemantics.md",
        "title": "HED Annotation Semantics",
        "source_url": (
            "https://raw.githubusercontent.com/hed-standard/"
            "hed-resources/main/docs/source/HedAnnotationSemantics.md"
        ),
        "public_url": ("https://www.hedtags.org/hed-resources/HedAnnotationSemantics.html"),
    },
    {
        "filename": "02_Terminology.md",
        "title": "HED Terminology",
        "source_url": (
            "https://raw.githubusercontent.com/hed-standard/"
            "hed-specification/main/docs/source/02_Terminology.md"
        ),
        "public_url": ("https://www.hedtags.org/hed-specification/02_Terminology.html"),
    },
]

OUTPUT_DIR = Path(__file__).parent.parent / "src" / "data" / "hed-docs"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def clean_myst_markdown(text: str) -> str:
    """Strip MyST/Sphinx directives from markdown, preserving content.

    Handles:
    - {admonition} blocks: extracts title and body, removes frontmatter
    - {list-table} blocks: converts to plain markdown table
    - {code-block} directives: converts to standard fenced code blocks
    - {index} directives: strips entirely
    - (anchor-name)= lines: strips entirely
    """
    lines = text.split("\n")
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Strip anchor definitions: (anchor-name)=
        if re.match(r"^\([\w-]+\)=\s*$", line):
            i += 1
            continue

        # Handle admonition blocks (``` or ```` or ````` prefixed)
        admonition_match = re.match(r"^(`{3,})\{admonition\}\s*(.*)", line)
        if admonition_match:
            fence = admonition_match.group(1)
            title = admonition_match.group(2).strip()
            i += 1

            # Skip frontmatter (--- delimited block)
            if i < len(lines) and lines[i].strip() == "---":
                i += 1
                while i < len(lines) and lines[i].strip() != "---":
                    i += 1
                if i < len(lines):
                    i += 1  # skip closing ---

            # Emit title as bold text (avoid double-bolding if already bold)
            if title:
                if "**" in title:
                    result.append(title)
                else:
                    result.append(f"**{title}**")
                result.append("")

            # Collect body until closing fence
            while i < len(lines):
                if lines[i].strip() == fence:
                    i += 1
                    break
                result.append(lines[i])
                i += 1

            result.append("")
            continue

        # Handle list-table blocks
        list_table_match = re.match(r"^(`{3,})\{list-table\}", line)
        if list_table_match:
            fence = list_table_match.group(1)
            i += 1

            # Skip frontmatter
            if i < len(lines) and lines[i].strip() == "---":
                i += 1
                while i < len(lines) and lines[i].strip() != "---":
                    i += 1
                if i < len(lines):
                    i += 1

            # Parse list-table rows into markdown table
            rows = []
            current_row = []
            current_cell_lines = []

            while i < len(lines):
                if lines[i].strip() == fence:
                    i += 1
                    break

                stripped = lines[i].rstrip()
                if stripped.startswith("* - "):
                    # New row, first cell
                    if current_row or current_cell_lines:
                        if current_cell_lines:
                            current_row.append(" ".join(current_cell_lines).strip())
                        if current_row:
                            rows.append(current_row)
                    current_row = [stripped[4:].strip()]
                    current_cell_lines = []
                elif stripped.startswith("  - "):
                    # New cell in current row
                    if current_cell_lines:
                        current_row.append(" ".join(current_cell_lines).strip())
                        current_cell_lines = []
                    current_row.append(stripped[4:].strip())
                elif stripped.startswith("    "):
                    # Continuation of current cell
                    current_cell_lines.append(stripped.strip())
                i += 1

            # Flush last row
            if current_cell_lines:
                current_row.append(" ".join(current_cell_lines).strip())
            if current_row:
                rows.append(current_row)

            # Emit as markdown table
            if rows:
                header = rows[0]
                col_count = len(header)
                result.append("| " + " | ".join(header) + " |")
                result.append("| " + " | ".join(["---"] * col_count) + " |")
                for row in rows[1:]:
                    # Pad row to match header columns
                    padded = row + [""] * (col_count - len(row))
                    result.append("| " + " | ".join(padded[:col_count]) + " |")
                result.append("")

            continue

        # Handle code-block directives
        code_block_match = re.match(r"^(`{3,})\{code-block\}\s*(.*)", line)
        if code_block_match:
            fence = code_block_match.group(1)
            lang = code_block_match.group(2).strip()
            i += 1

            # Skip frontmatter
            if i < len(lines) and lines[i].strip() == "---":
                i += 1
                while i < len(lines) and lines[i].strip() != "---":
                    i += 1
                if i < len(lines):
                    i += 1

            result.append(f"```{lang}")
            while i < len(lines):
                if lines[i].strip() == fence:
                    i += 1
                    break
                result.append(lines[i])
                i += 1
            result.append("```")
            result.append("")
            continue

        # Handle index directives (strip entirely)
        index_match = re.match(r"^(`{3,})\{index\}", line)
        if index_match:
            fence = index_match.group(1)
            i += 1
            while i < len(lines):
                if lines[i].strip() == fence:
                    i += 1
                    break
                i += 1
            continue

        # Regular line: pass through
        result.append(line)
        i += 1

    return "\n".join(result)


def sha256_hash(content: str) -> str:
    """Compute SHA-256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_manifest() -> dict:
    """Load existing manifest or return empty structure."""
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"docs": [], "schema_version": "1.0"}


def save_manifest(manifest: dict) -> None:
    """Save manifest to disk."""
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def fetch_and_process(force: bool = False) -> bool:
    """Fetch official HED docs, process, and save.

    Returns True if any docs were updated.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    existing_hashes = {d["filename"]: d.get("sha256", "") for d in manifest.get("docs", [])}

    updated = False
    new_docs = []

    for doc_info in DOCS:
        filename = doc_info["filename"]
        source_url = doc_info["source_url"]
        output_path = OUTPUT_DIR / filename

        print(f"Fetching {filename} from {source_url}...")
        response = httpx.get(source_url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        raw_content = response.text
        processed = clean_myst_markdown(raw_content)
        content_hash = sha256_hash(processed)

        if not force and content_hash == existing_hashes.get(filename):
            print(f"  {filename}: unchanged (hash match)")
            # Keep existing manifest entry
            for d in manifest.get("docs", []):
                if d["filename"] == filename:
                    new_docs.append(d)
                    break
            continue

        output_path.write_text(processed, encoding="utf-8")
        print(f"  {filename}: updated ({len(processed)} chars)")
        updated = True

        new_docs.append(
            {
                "filename": filename,
                "title": doc_info["title"],
                "source_url": source_url,
                "public_url": doc_info["public_url"],
                "fetched_at": datetime.now(UTC).isoformat(),
                "sha256": content_hash,
            }
        )

    manifest["docs"] = new_docs
    save_manifest(manifest)

    return updated


def main() -> None:
    force = "--force" in sys.argv

    try:
        updated = fetch_and_process(force=force)
    except httpx.HTTPError as e:
        print(f"Error fetching docs: {e}", file=sys.stderr)
        sys.exit(1)

    if updated:
        print("\nDocs updated. Commit the changes in src/data/hed-docs/")
    else:
        print("\nAll docs up to date.")


if __name__ == "__main__":
    main()
