/**
 * HED-BOT Cloudflare Worker (Proxy Mode)
 *
 * This worker acts as a caching proxy to the Python FastAPI backend,
 * which has all the strong prompts, real HED validation, and multi-agent workflow.
 */

// Worker configuration
const CONFIG = {
  CACHE_TTL: 3600, // 1 hour cache for identical requests
  RATE_LIMIT_PER_MINUTE: 20,
  REQUEST_TIMEOUT: 120000, // 2 minutes for long-running annotation workflows
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
        return await handleHealth(request, env, corsHeaders);
      } else if (url.pathname === '/version') {
        return await handleVersion(request, env, corsHeaders);
      } else if (url.pathname === '/annotate' && request.method === 'POST') {
        return await handleAnnotate(request, env, ctx, corsHeaders);
      } else if (url.pathname === '/annotate-from-image' && request.method === 'POST') {
        return await handleAnnotateFromImage(request, env, corsHeaders);
      } else if (url.pathname === '/validate' && request.method === 'POST') {
        return await handleValidate(request, env, corsHeaders);
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
 * Root endpoint
 */
function handleRoot(corsHeaders) {
  return new Response(JSON.stringify({
    name: 'HED-BOT API (Cloudflare Workers Proxy)',
    version: '2.0.0',
    description: 'Proxy to Python FastAPI backend with caching and rate limiting',
    mode: 'proxy',
    endpoints: {
      'POST /annotate': 'Generate HED annotation from description',
      'POST /annotate-from-image': 'Generate HED annotation from image',
      'POST /validate': 'Validate HED annotation string',
      'GET /health': 'Health check',
      'GET /version': 'Get API version',
    },
  }), {
    headers: { ...corsHeaders, 'Content-Type': 'application/json' },
  });
}

/**
 * Version endpoint (proxies to backend to get actual version)
 */
async function handleVersion(request, env, corsHeaders) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({
      version: 'unknown',
      error: 'Backend not configured',
    }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    // Get version from backend
    const response = await fetch(`${backendUrl}/version`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });

    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }

    const version = await response.json();

    return new Response(JSON.stringify(version), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({
      version: 'unknown',
      error: error.message,
    }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
}

/**
 * Health check endpoint
 */
async function handleHealth(request, env, corsHeaders) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({
      status: 'error',
      message: 'BACKEND_URL not configured',
    }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    // Check backend health
    const response = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(5000), // 5 second timeout
    });

    const backendHealth = await response.json();

    return new Response(JSON.stringify({
      status: 'healthy',
      proxy: 'operational',
      backend: backendHealth,
      backend_url: backendUrl,
    }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({
      status: 'degraded',
      proxy: 'operational',
      backend: 'unreachable',
      error: error.message,
    }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
}

/**
 * Hash a string using SHA-256
 */
async function hashString(str) {
  const encoder = new TextEncoder();
  const data = encoder.encode(str);
  const hashBuffer = await crypto.subtle.digest('SHA-256', data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Main annotation endpoint (proxies to backend)
 */
async function handleAnnotate(request, env, ctx, corsHeaders) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({ error: 'Backend not configured' }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

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

  // Check cache - hash the description to avoid KV key length limit (512 bytes)
  const descriptionHash = await hashString(description);
  const cacheKey = `hed:${schema_version}:${descriptionHash}`;
  const cached = await env.HED_CACHE?.get(cacheKey, 'json');
  if (cached) {
    return new Response(JSON.stringify({ ...cached, cached: true }), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/annotate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        description,
        schema_version,
        max_validation_attempts,
        run_assessment,
      }),
      signal: AbortSignal.timeout(CONFIG.REQUEST_TIMEOUT),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Backend error: ${error}`);
    }

    const result = await response.json();

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
      error: 'Backend request failed',
      details: error.message,
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
}

/**
 * Image annotation endpoint (proxies to backend)
 */
async function handleAnnotateFromImage(request, env, corsHeaders) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({ error: 'Backend not configured' }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    const body = await request.json();
    const {
      image,
      prompt,
      schema_version = '8.4.0',
      max_validation_attempts = 5,
      run_assessment = false,
    } = body;

    if (!image || image.trim() === '') {
      return new Response(JSON.stringify({ error: 'Image is required' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/annotate-from-image`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image,
        prompt,
        schema_version,
        max_validation_attempts,
        run_assessment,
      }),
      signal: AbortSignal.timeout(CONFIG.REQUEST_TIMEOUT),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Backend error: ${error}`);
    }

    const result = await response.json();

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Image annotation request failed',
      details: error.message,
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
}

/**
 * Validation endpoint (proxies to backend)
 */
async function handleValidate(request, env, corsHeaders) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({ error: 'Backend not configured' }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    const body = await request.json();

    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000), // 30 second timeout
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Backend error: ${error}`);
    }

    const result = await response.json();

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Validation request failed',
      details: error.message,
    }), {
      status: 500,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }
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
