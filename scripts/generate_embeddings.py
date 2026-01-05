#!/usr/bin/env python
"""Generate pre-computed embeddings for HED schema tags.

This script generates embeddings for all tags in the HED schema and saves them
to a JSON file for fast loading at runtime. It also generates embeddings for
the keyword index entries.

Usage:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --schema-version 8.4.0
    python scripts/generate_embeddings.py --output data/tag-embeddings.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_SCHEMA_VERSION = "8.4.0"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "tag-embeddings.json"


def load_hed_schema(version: str) -> tuple[list[dict], list[str]]:
    """Load HED schema and extract all tags.

    Args:
        version: HED schema version (e.g., "8.4.0")

    Returns:
        Tuple of (tag_entries, extendable_tags)
        Each tag_entry is a dict with tag, long_form, prefix
    """
    from hed import load_schema_version

    logger.info(f"Loading HED schema version {version}...")
    schema = load_schema_version(xml_version=version)

    tags = []
    extendable = []

    for _name, entry in schema.tags.items():
        tag_info = {
            "tag": entry.short_tag_name,
            "long_form": entry.name if hasattr(entry, "name") else entry.short_tag_name,
            "prefix": "",  # Base schema has no prefix
        }
        tags.append(tag_info)

        # Check if tag allows extension
        if hasattr(entry, "has_attribute") and entry.has_attribute("extensionAllowed"):
            extendable.append(entry.short_tag_name)

    logger.info(f"Loaded {len(tags)} tags from schema {version}")
    return tags, extendable


def load_library_schemas(library_specs: list[str]) -> list[dict]:
    """Load library schemas and extract tags with prefixes.

    Args:
        library_specs: List of library specs (e.g., ["sc:score_2.1.0"])

    Returns:
        List of tag entries with library prefixes
    """
    from hed import load_schema_version

    all_tags = []

    for spec in library_specs:
        if ":" not in spec:
            logger.warning(f"Invalid library spec (missing prefix): {spec}")
            continue

        prefix, version = spec.split(":", 1)
        prefix = f"{prefix}:"

        try:
            logger.info(f"Loading library schema: {spec}...")
            schema = load_schema_version(xml_version=version)

            for _name, entry in schema.tags.items():
                tag_info = {
                    "tag": entry.short_tag_name,
                    "long_form": entry.name if hasattr(entry, "name") else entry.short_tag_name,
                    "prefix": prefix,
                }
                all_tags.append(tag_info)

            logger.info(f"Loaded {len(all_tags)} tags from {spec}")
        except Exception as e:
            logger.error(f"Failed to load library schema {spec}: {e}")

    return all_tags


def generate_tag_embeddings(
    tags: list[dict],
    model_id: str = DEFAULT_MODEL_ID,
) -> list[dict]:
    """Generate embeddings for all tags.

    Args:
        tags: List of tag entries (dict with tag, long_form, prefix)
        model_id: HuggingFace model ID

    Returns:
        List of tag entries with vector field added
    """
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_id}...")
    model = SentenceTransformer(model_id)
    logger.info("Model loaded successfully")

    # Prepare texts for embedding - use tag names
    texts = []
    for tag in tags:
        # Use short form tag name, with prefix context if present
        text = tag["tag"]
        if tag["prefix"]:
            text = f"{tag['prefix']}{tag['tag']}"
        texts.append(text.lower().replace("-", " "))

    logger.info(f"Generating embeddings for {len(texts)} tags...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    # Add vectors to tag entries
    result = []
    for i, tag in enumerate(tags):
        tag_with_vector = tag.copy()
        tag_with_vector["vector"] = embeddings[i].tolist()
        result.append(tag_with_vector)

    return result


def generate_keyword_embeddings(
    model_id: str = DEFAULT_MODEL_ID,
) -> list[dict]:
    """Generate embeddings for keyword index entries.

    Args:
        model_id: HuggingFace model ID

    Returns:
        List of keyword entries with vector field
    """
    from sentence_transformers import SentenceTransformer

    from src.utils.semantic_search import KEYWORD_INDEX

    logger.info(f"Loading embedding model for keywords: {model_id}...")
    model = SentenceTransformer(model_id)

    keywords = list(KEYWORD_INDEX.keys())
    logger.info(f"Generating embeddings for {len(keywords)} keywords...")

    embeddings = model.encode(
        [k.replace("-", " ") for k in keywords],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    result = []
    for i, keyword in enumerate(keywords):
        result.append(
            {
                "keyword": keyword,
                "targets": KEYWORD_INDEX[keyword],
                "vector": embeddings[i].tolist(),
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate HED tag embeddings for semantic search")
    parser.add_argument(
        "--schema-version",
        default=DEFAULT_SCHEMA_VERSION,
        help=f"HED schema version (default: {DEFAULT_SCHEMA_VERSION})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Embedding model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--libraries",
        nargs="*",
        default=[],
        help="Library schemas to include (e.g., sc:score_2.1.0 la:lang_1.1.0)",
    )
    parser.add_argument(
        "--skip-keywords",
        action="store_true",
        help="Skip generating keyword embeddings",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load base schema
    base_tags, extendable = load_hed_schema(args.schema_version)

    # Load library schemas
    library_tags = load_library_schemas(args.libraries) if args.libraries else []

    # Combine all tags
    all_tags = base_tags + library_tags

    # Generate embeddings
    tag_embeddings = generate_tag_embeddings(all_tags, args.model)

    # Generate keyword embeddings
    keyword_embeddings = []
    if not args.skip_keywords:
        keyword_embeddings = generate_keyword_embeddings(args.model)

    # Get embedding dimensions
    dimensions = len(tag_embeddings[0]["vector"]) if tag_embeddings else 1024

    # Save to file
    output_data = {
        "version": "1.0.0",
        "model_id": args.model,
        "schema_version": args.schema_version,
        "library_schemas": args.libraries,
        "dimensions": dimensions,
        "tags": tag_embeddings,
        "keywords": keyword_embeddings,
    }

    logger.info(f"Saving embeddings to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(output_data, f)

    # Report size
    file_size = args.output.stat().st_size / (1024 * 1024)
    logger.info(f"Done! Output file size: {file_size:.2f} MB")
    logger.info(f"  - Tags: {len(tag_embeddings)}")
    logger.info(f"  - Keywords: {len(keyword_embeddings)}")
    logger.info(f"  - Dimensions: {dimensions}")


if __name__ == "__main__":
    main()
