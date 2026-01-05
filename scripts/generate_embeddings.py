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
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data"
# Library schemas to include by default (SCORE for clinical/EEG, LANG for language)
DEFAULT_LIBRARIES = ["sc:score_2.1.0", "la:lang_1.1.0"]


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
        # Get full hierarchical path (e.g., "Event/Sensory-event/Visual-presentation")
        long_form = entry.long_tag_name if hasattr(entry, "long_tag_name") else entry.short_tag_name
        tag_info = {
            "tag": entry.short_tag_name,
            "long_form": long_form,
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
        count_before = len(all_tags)

        try:
            logger.info(f"Loading library schema: {spec}...")
            schema = load_schema_version(xml_version=version)

            for _name, entry in schema.tags.items():
                # Get full hierarchical path with prefix
                long_form = (
                    entry.long_tag_name if hasattr(entry, "long_tag_name") else entry.short_tag_name
                )
                tag_info = {
                    "tag": entry.short_tag_name,
                    "long_form": f"{prefix}{long_form}",  # Include prefix in long form
                    "prefix": prefix,
                }
                all_tags.append(tag_info)

            logger.info(f"Loaded {len(all_tags) - count_before} tags from {spec}")
        except Exception as e:
            logger.error(f"Failed to load library schema {spec}: {e}")

    return all_tags


def expand_tag_path(path: str) -> str:
    """Expand a HED tag path into a more readable format for embeddings.

    Converts camelCase and hyphens to spaces for better semantic matching.
    E.g., "Event/Sensory-event/Visual-presentation" -> "event sensory event visual presentation"

    Args:
        path: HED tag path (possibly with prefix like "sc:")

    Returns:
        Space-separated lowercase string
    """
    import re

    # Expand camelCase: "SensoryEvent" -> "Sensory Event"
    path = re.sub(r"([a-z])([A-Z])", r"\1 \2", path)
    # Replace hyphens and slashes with spaces
    path = path.replace("-", " ").replace("/", " ")
    # Remove prefix colons (sc:, la:)
    path = re.sub(r"^\w+:", "", path)
    return path.lower()


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

    # Prepare texts for embedding - use SHORT TAG NAME (not full path)
    # Full hierarchy paths don't produce good similarity matches
    # e.g., "Visual-presentation" -> "visual presentation"
    texts = []
    for tag in tags:
        # Use short tag name expanded (camelCase -> spaces, hyphens -> spaces)
        text = expand_tag_path(tag["tag"])
        texts.append(text)

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


def save_embeddings_file(
    output_path: Path,
    embeddings: list[dict],
    model_id: str,
    schema_spec: str,
    embed_type: str = "tags",
) -> None:
    """Save embeddings to a JSON file.

    Args:
        output_path: Path to output file
        embeddings: List of embedding entries
        model_id: Model ID used for embeddings
        schema_spec: Schema specification (e.g., "8.4.0" or "sc:score_2.1.0")
        embed_type: Type of embeddings ("tags" or "keywords")
    """
    dimensions = len(embeddings[0]["vector"]) if embeddings else 1024

    output_data = {
        "version": "1.0.0",
        "model_id": model_id,
        "schema": schema_spec,
        "type": embed_type,
        "dimensions": dimensions,
        "count": len(embeddings),
        "embeddings": embeddings,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f)

    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved {output_path.name}: {len(embeddings)} {embed_type}, {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate modular HED tag embeddings for semantic search"
    )
    parser.add_argument(
        "--schema-version",
        default=DEFAULT_SCHEMA_VERSION,
        help=f"HED base schema version (default: {DEFAULT_SCHEMA_VERSION})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"Embedding model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--libraries",
        nargs="*",
        default=DEFAULT_LIBRARIES,
        help=f"Library schemas to include (default: {DEFAULT_LIBRARIES})",
    )
    parser.add_argument(
        "--no-libraries",
        action="store_true",
        help="Skip library schemas (only generate base schema embeddings)",
    )
    parser.add_argument(
        "--skip-keywords",
        action="store_true",
        help="Skip generating keyword embeddings",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Generate single combined file instead of modular files",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which libraries to process
    libraries = [] if args.no_libraries else args.libraries

    # Load the embedding model once
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {args.model}...")
    model = SentenceTransformer(args.model)
    logger.info("Model loaded successfully\n")

    total_tags = 0
    all_files = []

    # Generate base schema embeddings
    logger.info(f"=== Base Schema: {args.schema_version} ===")
    base_tags, _extendable = load_hed_schema(args.schema_version)

    texts = [expand_tag_path(tag["tag"]) for tag in base_tags]
    logger.info(f"Generating embeddings for {len(texts)} tags...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    base_embeddings = []
    for i, tag in enumerate(base_tags):
        tag_with_vector = tag.copy()
        tag_with_vector["vector"] = embeddings[i].tolist()
        base_embeddings.append(tag_with_vector)

    base_output = args.output_dir / f"embeddings-base-{args.schema_version}.json"
    save_embeddings_file(base_output, base_embeddings, args.model, args.schema_version)
    total_tags += len(base_embeddings)
    all_files.append(base_output)

    # Generate library schema embeddings (one file per library)
    for lib_spec in libraries:
        logger.info(f"\n=== Library Schema: {lib_spec} ===")
        lib_tags = load_library_schemas([lib_spec])

        if not lib_tags:
            logger.warning(f"No tags loaded for {lib_spec}, skipping")
            continue

        texts = [expand_tag_path(tag["tag"]) for tag in lib_tags]
        logger.info(f"Generating embeddings for {len(texts)} tags...")
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

        lib_embeddings = []
        for i, tag in enumerate(lib_tags):
            tag_with_vector = tag.copy()
            tag_with_vector["vector"] = embeddings[i].tolist()
            lib_embeddings.append(tag_with_vector)

        # Create filename from lib spec: "sc:score_2.1.0" -> "embeddings-sc-score_2.1.0.json"
        lib_filename = lib_spec.replace(":", "-")
        lib_output = args.output_dir / f"embeddings-{lib_filename}.json"
        save_embeddings_file(lib_output, lib_embeddings, args.model, lib_spec)
        total_tags += len(lib_embeddings)
        all_files.append(lib_output)

    # Generate keyword embeddings
    if not args.skip_keywords:
        logger.info("\n=== Keywords ===")
        keyword_embeddings = generate_keyword_embeddings(args.model)
        kw_output = args.output_dir / "embeddings-keywords.json"
        save_embeddings_file(kw_output, keyword_embeddings, args.model, "keywords", "keywords")
        all_files.append(kw_output)

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total tags: {total_tags}")
    logger.info("Files generated:")
    total_size = 0
    for f in all_files:
        size = f.stat().st_size / (1024 * 1024)
        total_size += size
        logger.info(f"  - {f.name}: {size:.2f} MB")
    logger.info(f"Total size: {total_size:.2f} MB")


if __name__ == "__main__":
    main()
