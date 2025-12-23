/**
 * HEDit Cloudflare Worker (Proxy Mode)
 *
 * This worker acts as a caching proxy to the Python FastAPI backend,
 * which has all the strong prompts, real HED validation, and multi-agent workflow.
 */

// Worker configuration (environment-aware)
function getConfig(env) {
  const isDev = env.ENVIRONMENT === 'development';
  return {
    CACHE_TTL: isDev ? 300 : 3600, // 5 min for dev, 1 hour for prod
    RATE_LIMIT_PER_MINUTE: isDev ? 60 : 20, // Higher limit for dev testing
    REQUEST_TIMEOUT: 120000, // 2 minutes for long-running annotation workflows
    ALLOWED_ORIGIN: 'https://hedit.pages.dev', // Production frontend
    TURNSTILE_VERIFY_URL: 'https://challenges.cloudflare.com/turnstile/v0/siteverify',
    IS_DEV: isDev,
  };
}

/**
 * Verify Cloudflare Turnstile token
 * @param {string} token - The Turnstile response token from the client
 * @param {string} secretKey - The Turnstile secret key
 * @param {string} ip - The client's IP address
 * @returns {Promise<{success: boolean, error?: string}>}
 */
async function verifyTurnstileToken(token, secretKey, ip) {
  if (!token) {
    return { success: false, error: 'Missing Turnstile token' };
  }

  if (!secretKey) {
    // If no secret key configured, skip verification (for development)
    console.warn('TURNSTILE_SECRET_KEY not configured, skipping verification');
    return { success: true };
  }

  try {
    const formData = new URLSearchParams();
    formData.append('secret', secretKey);
    formData.append('response', token);
    if (ip) {
      formData.append('remoteip', ip);
    }

    const response = await fetch('https://challenges.cloudflare.com/turnstile/v0/siteverify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: formData,
    });

    const result = await response.json();

    if (result.success) {
      return { success: true };
    } else {
      return {
        success: false,
        error: `Turnstile verification failed: ${result['error-codes']?.join(', ') || 'Unknown error'}`,
      };
    }
  } catch (error) {
    return { success: false, error: `Turnstile verification error: ${error.message}` };
  }
}

export default {
  async fetch(request, env, ctx) {
    const CONFIG = getConfig(env);
    const origin = request.headers.get('Origin');

    // CORS validation - allow hedit.pages.dev, hed-bot.pages.dev (legacy), annotation.garden, and preview deployments
    const isAllowedOrigin = origin === CONFIG.ALLOWED_ORIGIN ||
                           origin === 'https://hedit.pages.dev' || // Production (new)
                           origin?.endsWith('.hedit.pages.dev') || // Preview/develop deployments (new)
                           origin === 'https://hed-bot.pages.dev' || // Production (legacy, until migration)
                           origin?.endsWith('.hed-bot.pages.dev') || // Preview/develop deployments (legacy)
                           origin === 'https://annotation.garden' || // Main AGI site
                           origin?.endsWith('.annotation.garden') || // AGI subdomains
                           origin?.startsWith('http://localhost:'); // Allow localhost for dev

    // CORS headers
    // Include BYOK headers (X-OpenRouter-*) for CLI and programmatic access
    const corsHeaders = {
      'Access-Control-Allow-Origin': isAllowedOrigin ? origin : CONFIG.ALLOWED_ORIGIN,
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-API-Key, X-OpenRouter-Key, X-OpenRouter-Model, X-OpenRouter-Provider, X-OpenRouter-Temperature, X-OpenRouter-Eval-Model, X-OpenRouter-Eval-Provider, X-User-Id',
      'Access-Control-Allow-Credentials': 'true',
    };

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const url = new URL(request.url);

      // Route requests
      if (url.pathname === '/health') {
        return await handleHealth(request, env, corsHeaders, CONFIG);
      } else if (url.pathname === '/version') {
        return await handleVersion(request, env, corsHeaders);
      } else if (url.pathname === '/annotate' && request.method === 'POST') {
        return await handleAnnotate(request, env, ctx, corsHeaders, CONFIG);
      } else if (url.pathname === '/annotate-from-image' && request.method === 'POST') {
        return await handleAnnotateFromImage(request, env, corsHeaders, CONFIG);
      } else if (url.pathname === '/validate' && request.method === 'POST') {
        return await handleValidate(request, env, corsHeaders, CONFIG);
      } else if (url.pathname === '/feedback' && request.method === 'POST') {
        return await handleFeedback(request, env, corsHeaders, CONFIG);
      } else if (url.pathname === '/') {
        return handleRoot(corsHeaders, CONFIG);
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
function handleRoot(corsHeaders, CONFIG) {
  return new Response(JSON.stringify({
    name: 'HEDit API (Cloudflare Workers Proxy)',
    version: '2.0.0',
    description: 'Proxy to Python FastAPI backend with caching and rate limiting',
    mode: 'proxy',
    environment: CONFIG.IS_DEV ? 'development' : 'production',
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
async function handleHealth(request, env, corsHeaders, CONFIG) {
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
      environment: CONFIG.IS_DEV ? 'development' : 'production',
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
async function handleAnnotate(request, env, ctx, corsHeaders, CONFIG) {
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
    cf_turnstile_response, // Turnstile token from frontend
  } = body;

  if (!description || description.trim() === '') {
    return new Response(JSON.stringify({ error: 'Description is required' }), {
      status: 400,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  // Check for BYOK (Bring Your Own Key) mode - CLI/programmatic access with user's own API key
  // BYOK users skip Turnstile verification since:
  // 1. They can't complete Turnstile challenges (CLI/programmatic access)
  // 2. They're using their own API key, so any abuse is on their own account
  const isBYOK = request.headers.get('X-OpenRouter-Key') !== null;

  // Verify Turnstile token (required for non-BYOK requests in production)
  if (!isBYOK) {
    const clientIp = request.headers.get('CF-Connecting-IP');
    const turnstileResult = await verifyTurnstileToken(
      cf_turnstile_response,
      env.TURNSTILE_SECRET_KEY,
      clientIp
    );

    if (!turnstileResult.success) {
      return new Response(JSON.stringify({
        error: 'Bot verification failed',
        details: turnstileResult.error,
      }), {
        status: 403,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  }

  // Check rate limit
  if (!await checkRateLimit(request, env, CONFIG)) {
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
    // Prepare headers for backend request
    const backendHeaders = {
      'Content-Type': 'application/json',
    };

    // Add backend API key if configured
    if (env.BACKEND_API_KEY) {
      backendHeaders['X-API-Key'] = env.BACKEND_API_KEY;
    }

    // Forward BYOK and user ID headers to backend
    const byokHeaders = ['X-OpenRouter-Key', 'X-OpenRouter-Model', 'X-OpenRouter-Provider', 'X-OpenRouter-Temperature', 'X-OpenRouter-Eval-Model', 'X-OpenRouter-Eval-Provider', 'X-User-Id'];
    for (const header of byokHeaders) {
      const value = request.headers.get(header);
      if (value) {
        backendHeaders[header] = value;
      }
    }

    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/annotate`, {
      method: 'POST',
      headers: backendHeaders,
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
async function handleAnnotateFromImage(request, env, corsHeaders, CONFIG) {
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
      cf_turnstile_response, // Turnstile token from frontend
    } = body;

    if (!image || image.trim() === '') {
      return new Response(JSON.stringify({ error: 'Image is required' }), {
        status: 400,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }

    // Check for BYOK (Bring Your Own Key) mode - CLI/programmatic access with user's own API key
    const isBYOK = request.headers.get('X-OpenRouter-Key') !== null;

    // Verify Turnstile token (required for non-BYOK requests in production)
    if (!isBYOK) {
      const clientIp = request.headers.get('CF-Connecting-IP');
      const turnstileResult = await verifyTurnstileToken(
        cf_turnstile_response,
        env.TURNSTILE_SECRET_KEY,
        clientIp
      );

      if (!turnstileResult.success) {
        return new Response(JSON.stringify({
          error: 'Bot verification failed',
          details: turnstileResult.error,
        }), {
          status: 403,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        });
      }
    }

    // Prepare headers for backend request
    const backendHeaders = {
      'Content-Type': 'application/json',
    };

    // Add backend API key if configured
    if (env.BACKEND_API_KEY) {
      backendHeaders['X-API-Key'] = env.BACKEND_API_KEY;
    }

    // Forward BYOK and user ID headers to backend
    const byokHeaders = ['X-OpenRouter-Key', 'X-OpenRouter-Model', 'X-OpenRouter-Provider', 'X-OpenRouter-Temperature', 'X-OpenRouter-Eval-Model', 'X-OpenRouter-Eval-Provider', 'X-User-Id'];
    for (const header of byokHeaders) {
      const value = request.headers.get(header);
      if (value) {
        backendHeaders[header] = value;
      }
    }

    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/annotate-from-image`, {
      method: 'POST',
      headers: backendHeaders,
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
async function handleValidate(request, env, corsHeaders, CONFIG) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({ error: 'Backend not configured' }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    const body = await request.json();

    // Prepare headers for backend request
    const backendHeaders = {
      'Content-Type': 'application/json',
    };

    // Add API key if configured
    if (env.BACKEND_API_KEY) {
      backendHeaders['X-API-Key'] = env.BACKEND_API_KEY;
    }

    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/validate`, {
      method: 'POST',
      headers: backendHeaders,
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
 * Feedback endpoint (proxies to backend)
 * This endpoint is public (no Turnstile required) to allow feedback submission
 */
async function handleFeedback(request, env, corsHeaders, CONFIG) {
  const backendUrl = env.BACKEND_URL;

  if (!backendUrl) {
    return new Response(JSON.stringify({ error: 'Backend not configured' }), {
      status: 503,
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });
  }

  try {
    const body = await request.json();

    // Prepare headers for backend request
    const backendHeaders = {
      'Content-Type': 'application/json',
    };

    // Add API key if configured (feedback endpoint may not require it, but include for consistency)
    if (env.BACKEND_API_KEY) {
      backendHeaders['X-API-Key'] = env.BACKEND_API_KEY;
    }

    // Proxy request to Python backend
    const response = await fetch(`${backendUrl}/feedback`, {
      method: 'POST',
      headers: backendHeaders,
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
      error: 'Feedback submission failed',
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
async function checkRateLimit(request, env, CONFIG) {
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
