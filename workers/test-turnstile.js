/**
 * Turnstile tests for the edge Worker (issue #181).
 *
 * Runs the Worker's real fetch() on requests that are rejected before any
 * network call: a missing token, and a production Worker without
 * TURNSTILE_SECRET_KEY, which must reject instead of skipping verification.
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

async function post(mount, path, env, body) {
  const request = new Request(`https://${EDGE_HOST}${mount}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Origin: 'https://annotation.garden' },
    body: JSON.stringify(body),
  });
  return worker.fetch(request, env, { waitUntil() {} });
}

for (const [path, body] of ENDPOINTS) {
  // Production without the secret rejects a request that carries a token
  let r = await post('/hedit', path, PROD_ENV, { ...body, cf_turnstile_response: 'token' });
  let details = r.status === 403 ? (await r.json()).details : null;
  check(details === 'Bot verification is not configured',
    `production without TURNSTILE_SECRET_KEY rejects ${path}`);

  // An unknown or missing ENVIRONMENT is treated as production
  r = await post('/hedit', path, { EDGE_HOST, MOUNT_PREFIX: '/hedit', BACKEND_URL }, { ...body, cf_turnstile_response: 'token' });
  check(r.status === 403, `a Worker without ENVIRONMENT rejects ${path} when the secret is missing`);

  // A request without a token is rejected in development too
  r = await post('/hedit-dev', path, DEV_ENV, body);
  details = r.status === 403 ? (await r.json()).details : null;
  check(details === 'Missing Turnstile token', `development still requires a token on ${path}`);
}

if (failures) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nAll Turnstile checks passed');
