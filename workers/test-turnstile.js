/**
 * Turnstile tests for the edge Worker (issue #181).
 *
 * Runs the Worker's real fetch(). A production Worker without
 * TURNSTILE_SECRET_KEY must reject instead of skipping verification; only
 * development may skip, and BYOK requests skip Turnstile by design. Requests
 * that pass Turnstile go on to a backend that does not resolve
 * (backend.invalid), so they end in a backend error, never a 403. Siteverify
 * is answered by a fixture standing in for Cloudflare's HTTP endpoint.
 * Run with: bun workers/test-turnstile.js
 */
import worker from './index.js';

const EDGE_HOST = 'edge.annotation.garden';
// Never contacted: every request here is rejected before the backend call
const BACKEND_URL = 'https://backend.invalid';
const PROD_ENV = { ENVIRONMENT: 'production', EDGE_HOST, MOUNT_PREFIX: '/hedit', BACKEND_URL };
const DEV_ENV = { ENVIRONMENT: 'development', EDGE_HOST, MOUNT_PREFIX: '/hedit-dev', BACKEND_URL };

// Every endpoint that verifies Turnstile, with the smallest body it accepts
const ENDPOINTS = [
  ['/annotate', { description: 'A red circle appears' }],
  ['/annotate/stream', { description: 'A red circle appears' }],
  ['/annotate-from-image', { image: 'data:image/png;base64,AAAA' }],
  ['/annotate-from-image/stream', { image: 'data:image/png;base64,AAAA' }],
];

let failures = 0;
function check(condition, message) {
  console.log(`${condition ? 'ok  ' : 'FAIL'} ${message}`);
  if (!condition) failures++;
}

async function post(mount, path, env, body, headers = {}) {
  const request = new Request(`https://${EDGE_HOST}${mount}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Origin: 'https://annotation.garden', ...headers },
    body: JSON.stringify(body),
  });
  return worker.fetch(request, env, { waitUntil() {} });
}

async function details(response) {
  return response.status === 403 ? (await response.json()).details : null;
}

// Siteverify fixture: answers Cloudflare's endpoint, passes everything else
// (the unresolvable backend) to the real fetch
const realFetch = globalThis.fetch;
let siteverify = null;
globalThis.fetch = (input, init) => {
  const url = typeof input === 'string' ? input : input.url;
  if (url.startsWith('https://challenges.cloudflare.com/')) {
    return siteverify();
  }
  return realFetch(input, init);
};
const siteverifyAnswers = (result) => () => Promise.resolve(Response.json(result));

for (const [path, body] of ENDPOINTS) {
  // Production without the secret rejects a request that carries a token
  let r = await post('/hedit', path, PROD_ENV, { ...body, cf_turnstile_response: 'token' });
  check(await details(r) === 'Bot verification is not configured',
    `production without TURNSTILE_SECRET_KEY rejects ${path}`);

  // An unknown or missing ENVIRONMENT is treated as production
  r = await post('/hedit', path, { EDGE_HOST, MOUNT_PREFIX: '/hedit', BACKEND_URL }, { ...body, cf_turnstile_response: 'token' });
  check(r.status === 403, `a Worker without ENVIRONMENT rejects ${path} when the secret is missing`);

  // A request without a token is rejected in development too
  r = await post('/hedit-dev', path, DEV_ENV, body);
  check(await details(r) === 'Missing Turnstile token', `development still requires a token on ${path}`);

  // Development without the secret skips verification and reaches the backend
  r = await post('/hedit-dev', path, DEV_ENV, { ...body, cf_turnstile_response: 'token' });
  check(r.status !== 403 && r.status >= 500, `development without the secret skips verification on ${path}`);

  // BYOK requests skip Turnstile, even on production without the secret
  r = await post('/hedit', path, PROD_ENV, body, { 'X-Anthropic-Key': 'user-key' });
  check(r.status !== 403 && r.status >= 500, `a BYOK request skips Turnstile on ${path}`);
}

// With the secret set, Siteverify decides
const WITH_SECRET = { ...PROD_ENV, TURNSTILE_SECRET_KEY: 'secret' };
const annotate = { description: 'A red circle appears', cf_turnstile_response: 'token' };

siteverify = siteverifyAnswers({ success: false, 'error-codes': ['invalid-input-response'] });
let r = await post('/hedit', '/annotate', WITH_SECRET, annotate);
check(await details(r) === 'Turnstile verification failed: invalid-input-response',
  'a token Siteverify rejects is a 403 that names the error code');

siteverify = () => Promise.reject(new TypeError('network down'));
r = await post('/hedit', '/annotate', WITH_SECRET, annotate);
check(await details(r) === 'Turnstile verification error: network down',
  'a Siteverify call that fails rejects the request');

siteverify = () => Promise.resolve(new Response('<html>', { status: 502 }));
r = await post('/hedit', '/annotate', WITH_SECRET, annotate);
check((await details(r))?.startsWith('Turnstile verification error:'),
  'a Siteverify answer that is not JSON rejects the request');

siteverify = siteverifyAnswers({ success: true });
r = await post('/hedit', '/annotate', WITH_SECRET, annotate);
check(r.status !== 403 && r.status >= 500, 'a token Siteverify accepts passes on to the backend');

globalThis.fetch = realFetch;

if (failures) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nAll Turnstile checks passed');
