# HED Validation Tools

## Overview
HED JavaScript validator provides comprehensive validation with detailed feedback beyond simple pass/fail.

## Key Tools

### String-Level Validation
```javascript
const { parseHedString, buildSchemasFromVersion } = require('hed-validator')

const schemas = await buildSchemasFromVersion('8.3.0')
const [parsed, errors, warnings] = parseHedString(hedString, schemas)
```

### Feedback Structure
Each issue contains:
- `internalCode`: Internal error code
- `hedCode`: HED specification code
- `level`: 'error' or 'warning'
- `message`: Human-readable description
- `parameters`: Contextual information (tag, line, file, etc.)

### Issue Processing
- `splitErrors()`: Categorize by severity
- `categorizeByCode()`: Group by issue type
- `reduceIssues()`: Summarize recurring issues

## Integration Strategy
1. Wrap validator in Python subprocess/API
2. Feed validation errors back to LLM for correction
3. Track validation attempts to prevent infinite loops
4. Cache schemas per session for performance

## Location
- Repository: `/Users/yahya/Documents/git/HED/hed-javascript`
- Main exports: `index.js`
