/**
 * HEDit frontend Worker
 *
 * Serves frontend/ as static assets at annotation.garden/hedit/ (dev:
 * annotation.garden/hedit-dev/). The product frontend lives on a path of the
 * organization's domain, so no visitor depends on an account-scoped
 * *.pages.dev name.
 *
 * Cloudflare delivers the full path, mount prefix included. The assets are
 * stored at the root of frontend/, so the prefix is stripped before the
 * lookup, and added back to any same-origin redirect the asset handler issues
 * (for example "/hedit/index.html" -> "/hedit/"), which would otherwise point
 * at the main annotation.garden site.
 */

// A mount prefix is exactly one path segment, such as "/hedit".
const MOUNT_PREFIX_PATTERN = /^\/[^/]+$/;

function misconfigured(message) {
  console.error(`Worker misconfigured: ${message}`);
  return new Response(`Worker misconfigured: ${message}`, { status: 500 });
}

/**
 * Re-add the mount prefix to a same-origin redirect from the asset handler.
 * Its Location is relative to the stripped path, so it never carries the
 * prefix itself.
 */
function remountRedirect(response, assetUrl, url, prefix) {
  const location = response.headers.get('Location');
  if (!location || response.status < 300 || response.status >= 400) {
    return response;
  }
  let target;
  try {
    target = new URL(location, assetUrl);
  } catch (error) {
    console.error(`Unparsable Location from asset handler: ${JSON.stringify(location)}`, error.stack);
    return response;
  }
  if (target.origin !== url.origin) {
    return response;
  }
  target.pathname = `${prefix}${target.pathname}`;
  const headers = new Headers(response.headers);
  headers.set('Location', target.toString());
  return new Response(response.body, { status: response.status, headers });
}

export default {
  async fetch(request, env) {
    const prefix = env.MOUNT_PREFIX;
    if (!MOUNT_PREFIX_PATTERN.test(prefix ?? '')) {
      return misconfigured(`MOUNT_PREFIX must be one path segment such as "/hedit", got ${JSON.stringify(prefix)}`);
    }
    if (!env.ASSETS) {
      return misconfigured('ASSETS binding missing (check [assets] in wrangler.toml)');
    }

    const url = new URL(request.url);
    // The frontend loads its assets by relative path, so it must be served
    // from "<prefix>/", never from the bare "<prefix>".
    if (url.pathname === prefix) {
      return Response.redirect(`${url.origin}${prefix}/${url.search}`, 308);
    }
    if (!url.pathname.startsWith(`${prefix}/`)) {
      console.warn(`Outside mount: ${request.method} ${url.hostname}${url.pathname} (this Worker serves ${prefix}/)`);
      return new Response(`Not Found: this Worker serves ${prefix}/ only`, { status: 404 });
    }

    const assetUrl = new URL(url);
    assetUrl.pathname = url.pathname.slice(prefix.length);
    let response;
    try {
      response = await env.ASSETS.fetch(new Request(assetUrl, request));
    } catch (error) {
      console.error(`Asset fetch failed: ${request.method} ${url.pathname}`, error.stack);
      return new Response('Bad Gateway: static assets unavailable', { status: 502 });
    }
    return remountRedirect(response, assetUrl, url, prefix);
  },
};
