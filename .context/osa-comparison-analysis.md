# HEDit vs OSA/HBD Comparison Analysis

## Status
As of issue #137, the documentation gap has been closed. HEDit now preloads
official HED docs (HedAnnotationSemantics.md, 02_Terminology.md) from bundled
files, matching OSA's approach. A weekly CI workflow keeps the docs current.

## Original Summary (historical)
The Open Science Assistant (OSA) HED assistant produces more thorough and accurate responses than HEDit. This analysis identifies the key differences and recommends improvements.

## Key Differences

### 1. Validation Approach

**HEDit (current):**
- Uses local JavaScript/Python validators via subprocess
- Validation happens as a workflow step (post-generation)
- No real-time self-check before showing results to users

**OSA:**
- Uses hedtools.org REST API for validation
- Has explicit "validate before showing examples" pattern in system prompt
- Agent self-checks every example before presenting to users
- No local installation dependency

**Recommendation:** Add hedtools.org API validation as a tool available to the annotation agent for self-checking. Keep local validators for the workflow validation step.

### 2. Tag Discovery (HED LSP)

**HEDit (current):**
- Relies entirely on LLM knowledge of HED tags
- Provides vocabulary list in system prompt (first 80 tags only)
- No semantic search capability

**OSA:**
- Has `suggest_hed_tags` tool using HED LSP semantic search
- Explicit 5-step workflow: identify concepts -> search -> select -> construct -> validate
- Graceful degradation if hed-lsp unavailable

**Recommendation:** Integrate `suggest_hed_tags` tool into HEDit's annotation agent. The LSP provides ground-truth tag lookup, reducing hallucinated tags.

### 3. System Prompt Engineering

**HEDit (current):**
- System prompt in `src/utils/hed_rules.py` focuses on rules + vocabulary
- Response format: "Provide ONLY the HED annotation string. NO explanations."
- No tool usage instructions (agents don't have tools beyond the workflow)
- Limited to first 80 vocabulary tags + first 20 extendable tags

**OSA:**
- System prompt has explicit tool usage sections with detailed workflows
- "Use tools proactively and liberally" philosophy
- CRITICAL validation workflow: generate -> validate -> fix if invalid -> never show invalid
- Tag discovery workflow: identify concepts -> search -> select -> construct -> validate
- Preloaded documentation in system prompt (~13k tokens of core HED docs)

**Recommendation:** Restructure system prompt to include tool usage workflows. Add preloaded core documentation. Remove the "NO explanations" constraint for interactive use.

### 4. Documentation Strategy

**HEDit (current):**
- HED rules hardcoded in Python constants (`hed_rules.py`)
- No on-demand documentation retrieval
- No access to official HED specification during annotation

**OSA:**
- 32 documentation sources in YAML config
- 2 preloaded (core semantics + terminology, ~13k tokens in system prompt)
- 30 on-demand via `retrieve_hed_docs` tool
- Categories: specification, introductory, quickstart, core, tools, advanced, reference, examples
- Links to hedtags.org for user reference

**Recommendation:** Add documentation retrieval capability. At minimum, preload HED annotation semantics and terminology into the system prompt.

### 5. Tool Architecture

**HEDit (current):**
- Agents are workflow nodes (no tools)
- Each agent is a function that takes state and returns updated state
- No runtime tool invocation within agents

**OSA:**
- Agents have bound tools (LangChain tool binding)
- 3 specialized tools: `validate_hed_string`, `suggest_hed_tags`, `get_hed_schema_versions`
- Auto-generated generic tools: `retrieve_docs`, `fetch_page_content`, `search_discussions`
- Agent can decide when to use tools during conversation

**Recommendation:** Add tool binding to the annotation agent. At minimum: validate_hed_string, suggest_hed_tags, get_hed_schema_versions.

### 6. Schema Version

**HEDit (current):** Defaults to HED 8.3.0
**OSA:** Defaults to HED 8.4.0

**Recommendation:** Update default to 8.4.0.

### 7. Error Handling and Feedback

**HEDit (current):**
- Good error augmentation via `error_remediation.py`
- Feedback summarizer condenses errors for LLM
- Multi-iteration refinement loop

**OSA:**
- Simpler error handling (hedtools.org returns structured errors)
- But more proactive: validates before showing, so fewer errors reach users
- Prevention > correction approach

**Recommendation:** Keep HEDit's error augmentation, but add proactive validation.

## Priority Improvements

### P0 (Critical)
1. **Add hedtools.org validation tool** - For self-checking before presenting results
2. **Add suggest_hed_tags tool** - HED LSP integration for better tag discovery
3. **Update default schema to 8.4.0**

### P1 (High)
4. **Preload core HED documentation** into annotation agent system prompt
5. **Add "validate before showing" workflow** to system prompt
6. **Add tag discovery workflow** to system prompt

### P2 (Medium)
7. **Add documentation retrieval tool** - On-demand access to 30+ HED docs
8. **Add schema versions tool** - Runtime schema version lookup
9. **Restructure system prompt** with tool usage sections

### P3 (Nice to have)
10. **Add GitHub knowledge tools** - Issue/PR search across HED repos
11. **Add paper search** - Citation discovery
12. **Widget embedding support** - Page context tool
