/**
 * HED-BOT Cloudflare Worker
 *
 * Serverless backend for HED annotation generation using OpenRouter/Cerebras.
 * Replaces FastAPI backend with fully serverless architecture.
 */

// Worker configuration
const CONFIG = {
  OPENROUTER_API_URL: 'https://openrouter.ai/api/v1/chat/completions',
  DEFAULT_MODEL: 'anthropic/claude-3.5-sonnet',
  CEREBRAS_MODEL: 'meta-llama/llama-3.3-70b-instruct',
  CACHE_TTL: 3600, // 1 hour cache for identical requests
  RATE_LIMIT_PER_MINUTE: 10,
};

export default {
  async fetch(request, env, ctx) {
    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const url = new URL(request.url);

      // Route requests
      if (url.pathname === '/health') {
        return handleHealth(corsHeaders);
      } else if (url.pathname === '/annotate' && request.method === 'POST') {
        return await handleAnnotate(request, env, ctx, corsHeaders);
      } else if (url.pathname === '/') {
        return handleRoot(corsHeaders);
      }

      return new Response('Not Found', { status: 404, headers: corsHeaders });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  },
};

/**
 * Health check endpoint
 */
function handleHealth(corsHeaders) {
  return new Response(JSON.stringify({
    status: 'healthy',
    version: '1.0.0',
    llm_available: true,
    validator_available: true,
  }), {
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

/**
 * Root endpoint
 */
function handleRoot(corsHeaders) {
  return new Response(JSON.stringify({
    name: 'HED-BOT API (Cloudflare Workers)',
    version: '1.0.0',
    description: 'Serverless HED annotation generation',
    endpoints: {
      'POST /annotate': 'Generate HED annotation from description',
      'GET /health': 'Health check',
    },
  }), {
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

/**
 * Main annotation endpoint
 */
async function handleAnnotate(request, env, ctx, corsHeaders) {
  const body = await request.json();
  const {
    description,
    schema_version = '8.4.0',
    max_validation_attempts = 3,
    run_assessment = false,
  } = body;

  if (!description || description.trim() === '') {
    return new Response(JSON.stringify({ error: 'Description is required' }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  // Check rate limit
  if (!await checkRateLimit(request, env)) {
    return new Response(JSON.stringify({ error: 'Rate limit exceeded' }), {
      status: 429,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  // Check cache
  const cacheKey = `hed:${schema_version}:${description}`;
  const cached = await env.HED_CACHE?.get(cacheKey, 'json');
  if (cached) {
    return new Response(JSON.stringify({ ...cached, cached: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    // Run annotation workflow
    const result = await runAnnotationWorkflow({
      description,
      schema_version,
      max_validation_attempts,
      run_assessment,
      api_key: env.OPENROUTER_API_KEY,
    });

    // Cache successful results
    if (result.is_valid && env.HED_CACHE) {
      ctx.waitUntil(
        env.HED_CACHE.put(cacheKey, JSON.stringify(result), {
          expirationTtl: CONFIG.CACHE_TTL,
        })
      );
    }

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Annotation workflow failed',
      details: error.message,
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
}

/**
 * Run complete annotation workflow
 */
async function runAnnotationWorkflow(config) {
  let annotation = '';
  let is_valid = false;
  let validation_attempts = 0;
  let validation_errors = [];
  let validation_warnings = [];
  let evaluation_feedback = '';
  let assessment_feedback = '';

  // Step 1: Generate initial annotation
  annotation = await generateAnnotation(config.description, config.api_key);
  validation_attempts++;

  // Step 2: Validate and refine (with retries)
  for (let attempt = 0; attempt < config.max_validation_attempts; attempt++) {
    const validation = await validateAnnotation(annotation, config.schema_version);

    if (validation.is_valid) {
      is_valid = true;
      validation_errors = validation.errors;
      validation_warnings = validation.warnings;
      break;
    }

    // If not last attempt, try to fix errors
    if (attempt < config.max_validation_attempts - 1) {
      annotation = await refineAnnotation(
        config.description,
        annotation,
        validation.errors,
        config.api_key
      );
      validation_attempts++;
    } else {
      validation_errors = validation.errors;
      validation_warnings = validation.warnings;
    }
  }

  // Step 3: Evaluate faithfulness (always run)
  evaluation_feedback = await evaluateAnnotation(
    config.description,
    annotation,
    config.api_key
  );
  const is_faithful = !evaluation_feedback.toLowerCase().includes('not faithful');

  // Step 4: Assess completeness (optional)
  let is_complete = true;
  if (config.run_assessment) {
    assessment_feedback = await assessAnnotation(
      config.description,
      annotation,
      config.api_key
    );
    is_complete = !assessment_feedback.toLowerCase().includes('incomplete');
  }

  return {
    annotation,
    is_valid,
    is_faithful,
    is_complete,
    validation_attempts,
    validation_errors,
    validation_warnings,
    evaluation_feedback,
    assessment_feedback,
    status: is_valid ? 'success' : 'failed',
  };
}

/**
 * Generate initial HED annotation
 */
async function generateAnnotation(description, apiKey) {
  const prompt = `You are an expert in HED (Hierarchical Event Descriptors) annotation.

Generate a valid HED annotation for the following event description:
"${description}"

Requirements:
1. Use HED 8.4.0 schema tags
2. Include timing information (Onset, Offset, Duration when applicable)
3. Use sensory modality tags (Visual, Auditory, etc.)
4. Include action tags (Press, Release, etc.)
5. Use parentheses for grouping related tags
6. Separate tags with commas
7. Use proper HED syntax

Provide ONLY the HED annotation string, no explanations.`;

  return await callOpenRouter(prompt, apiKey, true); // Use fast Cerebras model
}

/**
 * Refine annotation based on validation errors
 */
async function refineAnnotation(description, currentAnnotation, errors, apiKey) {
  const errorList = errors.join('\n');
  const prompt = `You are an expert in HED (Hierarchical Event Descriptors) annotation.

The following HED annotation has validation errors:
Annotation: ${currentAnnotation}

Errors:
${errorList}

Original description: "${description}"

Fix the annotation to resolve ALL errors while maintaining semantic accuracy.
Provide ONLY the corrected HED annotation string, no explanations.`;

  return await callOpenRouter(prompt, apiKey, true);
}

/**
 * Evaluate annotation faithfulness
 */
async function evaluateAnnotation(description, annotation, apiKey) {
  const prompt = `You are an expert evaluator of HED annotations.

Event description: "${description}"
HED annotation: ${annotation}

Evaluate if the annotation faithfully represents the event description.
Does it capture all important aspects? Is anything missing or incorrect?

Provide a brief assessment (2-3 sentences) stating whether it is faithful or not, and why.`;

  return await callOpenRouter(prompt, apiKey, false); // Use quality model
}

/**
 * Assess annotation completeness
 */
async function assessAnnotation(description, annotation, apiKey) {
  const prompt = `You are an expert assessor of HED annotation completeness.

Event description: "${description}"
HED annotation: ${annotation}

Assess the completeness of this annotation:
1. Are all sensory modalities captured?
2. Are temporal aspects properly described?
3. Are all actions and their properties included?
4. Are spatial relationships specified?

Provide a brief assessment stating whether it is complete or incomplete, and what might be missing.`;

  return await callOpenRouter(prompt, apiKey, false);
}

/**
 * Basic HED validation (simplified - checks syntax only)
 */
async function validateAnnotation(annotation, schemaVersion) {
  // Simplified validation - in production, you'd call HED validator API
  // or implement full validation logic

  const errors = [];
  const warnings = [];

  // Basic syntax checks
  if (!annotation || annotation.trim() === '') {
    errors.push('Empty annotation');
  }

  // Check for balanced parentheses
  let parenCount = 0;
  for (const char of annotation) {
    if (char === '(') parenCount++;
    if (char === ')') parenCount--;
    if (parenCount < 0) {
      errors.push('Unbalanced parentheses - closing before opening');
      break;
    }
  }
  if (parenCount > 0) {
    errors.push('Unbalanced parentheses - unclosed opening parenthesis');
  }

  // Check for basic HED structure (tags separated by commas)
  if (!annotation.includes(',') && !annotation.includes('(')) {
    warnings.push('Annotation may be too simple - consider adding more detail');
  }

  return {
    is_valid: errors.length === 0,
    errors,
    warnings,
  };
}

/**
 * Call OpenRouter API
 */
async function callOpenRouter(prompt, apiKey, useFastModel = true) {
  const model = useFastModel ? CONFIG.CEREBRAS_MODEL : CONFIG.DEFAULT_MODEL;

  const response = await fetch(CONFIG.OPENROUTER_API_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`,
      'HTTP-Referer': 'https://hed-bot.pages.dev',
      'X-Title': 'HED-BOT',
    },
    body: JSON.stringify({
      model,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
      temperature: 0.1,
      max_tokens: 1000,
    }),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenRouter API error: ${error}`);
  }

  const data = await response.json();
  return data.choices[0].message.content.trim();
}

/**
 * Rate limiting check
 */
async function checkRateLimit(request, env) {
  if (!env.RATE_LIMITER) return true; // No rate limiter configured

  const ip = request.headers.get('CF-Connecting-IP') || 'unknown';
  const key = `ratelimit:${ip}`;

  const current = await env.RATE_LIMITER.get(key);
  const count = current ? parseInt(current) : 0;

  if (count >= CONFIG.RATE_LIMIT_PER_MINUTE) {
    return false;
  }

  await env.RATE_LIMITER.put(key, (count + 1).toString(), {
    expirationTtl: 60,
  });

  return true;
}
