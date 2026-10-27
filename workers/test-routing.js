/**
 * Routing tests for the edge Worker's path mount (edge.annotation.garden/hedit).
 *
 * Runs the Worker's real fetch() on requests that resolve without a network
 * call: the root description, redirects, and requests outside the mount.
 * Run with: bun workers/test-routing.js
 */
import worker from './index.js';

const EDGE_HOST = 'edge.annotation.garden';
const PROD_ENV = { ENVIRONMENT: 'production', EDGE_HOST, MOUNT_PREFIX: '/hedit' };
const DEV_ENV = { ENVIRONMENT: 'development', EDGE_HOST, MOUNT_PREFIX: '/hedit-dev' };

let failures = 0;
function check(condition, message) {
  console.log(`${condition ? 'ok  ' : 'FAIL'} ${message}`);
  if (!condition) failures++;
}

async function call(url, env, init = {}) {
  return worker.fetch(new Request(url, init), env, { waitUntil() {} });
}

async function rootName(response) {
  return response.status === 200 ? (await response.json()).name : null;
}

// Mounted on the edge host: the prefix is stripped before routing
let r = await call('https://edge.annotation.garden/hedit/', PROD_ENV);
check(await rootName(r) === 'HEDit API (Cloudflare Workers Proxy)', 'prod /hedit/ reaches the root route');

r = await call('https://edge.annotation.garden/hedit-dev/', DEV_ENV);
check(await rootName(r) === 'HEDit API (Cloudflare Workers Proxy)', 'dev /hedit-dev/ reaches the root route');

// The bare mount path redirects to the trailing-slash form, keeping the query
r = await call('https://edge.annotation.garden/hedit?x=1', PROD_ENV);
check(r.status === 308 && r.headers.get('Location') === 'https://edge.annotation.garden/hedit/?x=1',
  'prod bare /hedit redirects to /hedit/ with the query string');

// Paths outside the mount are not routed
r = await call('https://edge.annotation.garden/annotate', PROD_ENV);
check(r.status === 404, 'unprefixed path on the edge host is 404');

r = await call('https://edge.annotation.garden/hedit-dev/', PROD_ENV);
check(r.status === 404 && (await r.text()).includes('serves /hedit/ only'),
  'prod Worker answers the dev mount with a 404 that names its own mount');

r = await call('https://edge.annotation.garden/heditfoo', PROD_ENV);
check(r.status === 404, '/heditfoo is outside the /hedit/ mount');

// Missing or malformed mount configuration fails loudly instead of routing everything to 404
r = await call('https://edge.annotation.garden/hedit/', { ENVIRONMENT: 'production', EDGE_HOST });
check(r.status === 500, 'missing MOUNT_PREFIX is a 500');

r = await call('https://edge.annotation.garden/hedit/', { ...PROD_ENV, MOUNT_PREFIX: '/hedit/' });
check(r.status === 500, 'MOUNT_PREFIX with a trailing slash is a 500');

r = await call('https://edge.annotation.garden/hedit/', { ...PROD_ENV, MOUNT_PREFIX: 'hedit' });
check(r.status === 500, 'MOUNT_PREFIX without a leading slash is a 500');

r = await call('https://hedit-api.example.workers.dev/', { ENVIRONMENT: 'production', MOUNT_PREFIX: '/hedit' });
check(r.status === 500, 'missing EDGE_HOST is a 500, on any host');

// Any other host (workers.dev, wrangler dev) keeps the raw paths
r = await call('https://hedit-api.example.workers.dev/', PROD_ENV);
check(await rootName(r) === 'HEDit API (Cloudflare Workers Proxy)', 'workers.dev root still routes unprefixed');

r = await call('https://hedit-api.example.workers.dev/hedit/', PROD_ENV);
check(r.status === 404, 'the mount prefix is not stripped on non-edge hosts (/hedit/ is 404 there)');

// CORS preflight answers the annotation.garden frontend on the mounted path
r = await call('https://edge.annotation.garden/hedit/annotate/stream', PROD_ENV, {
  method: 'OPTIONS',
  headers: { Origin: 'https://annotation.garden' },
});
check(r.headers.get('Access-Control-Allow-Origin') === 'https://annotation.garden',
  'preflight allows https://annotation.garden');

if (failures) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nAll routing checks passed');
