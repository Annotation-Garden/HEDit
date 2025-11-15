# HED Schema Information

## Current Version
- Latest Standard: 8.3.0
- Format: XML (primary), MediaWiki, TSV
- **Note**: JSON format not yet available - placeholder needed

## Schema Location
- Repository: `/Users/yahya/Documents/git/HED/hed-schemas`
- Latest XML: `standard_schema/hedxml/HEDLatest.xml`
- Library schemas: `library_schemas/` (SCORE, Lang, SLAM)

## Structure
- Hierarchical trees with top-level categories
- Child tags are "is-a" types of ancestors
- Short-form (e.g., `Square`) vs long-form (e.g., `Item/Object/Geometric-object/2D-shape/Rectangle/Square`)
- Multiple schemas can be combined with namespace prefixes

## Usage for Agents
1. Load schema into agent context (vocabulary constraints)
2. Provide schema hierarchy for tag selection
3. Cache schemas per session to reduce load time
4. Use for validation and vocabulary lookup

## Placeholder for JSON
Until schemas_latest_json is available, use XML parsing or wait for developer upload.
