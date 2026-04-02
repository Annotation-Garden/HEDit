# HED Schema Structure and Access

## Current Version
- **Standard schema**: HED 8.4.0 (HEDit default)
- **Library schemas**: HED_score, HED_lang, etc.

## Schema Loaders

HEDit has two schema loading systems:

### 1. JSON Schema Loader (`src/utils/json_schema_loader.py`)
- **Primary system** for annotation agent
- Loads from local directory or GitHub
- Extracts vocabulary (short-form tags only)
- Identifies `extensionAllowed` tags
- Provides closest-match suggestions for invalid tags
- Sources:
  - Local: HED schemas JSON directory
  - GitHub: `https://github.com/hed-standard/hed-schemas`

### 2. Legacy Schema Loader (`src/utils/schema_loader.py`)
- Uses HED Python library (`hed.schema.load_schema_version()`)
- Caches per session
- No extension metadata
- Used by Python validator fallback

## Schema Sources

### Local (preferred for Docker/production)
```
/Users/yahya/Documents/git/HED/hed-schemas/schemas_latest_json/
├── HEDLatest.json
├── HED_score_Latest.json
└── ...
```

### GitHub (fallback)
Automatic fetch from `hed-standard/hed-schemas` repository if local not available.

## Schema Structure
- Hierarchical tag tree with properties
- Short-form: Use tag name directly (e.g., `Red` not `Property/Informational-property/Sensory-property/Visual-attribute/Color/Red`)
- `extensionAllowed` attribute: Tags that can be extended with custom children
- Value classes: Numeric values with units (`Duration/# s`)

## Vocabulary Extraction
The JSON schema loader extracts:
- **All valid tags**: ~2000-3000 short-form tags
- **ExtensionAllowed tags**: Tags that can be extended (e.g., `Action`, `Label`)
- **Value format specs**: Expected formats for value tags

## Usage in Agents
- Annotation agent receives full vocabulary + extendable tags in system prompt
- Official HED docs (HedAnnotationSemantics.md, 02_Terminology.md) are preloaded
  from bundled files in src/data/hed-docs/ (fetched by scripts/fetch_hed_docs.py)
- ExtensionAllowed tags shown so agent knows what can be extended
