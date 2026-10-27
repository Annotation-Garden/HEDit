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
 * lookup, and added back to any redirect the asset handler issues (for
 * example "/hedit/index.html" -> "/hedit/"), which would otherwise point at
 * the main annotation.garden site.
 */
export default {
  async fetch(request, env) {
    const prefix = env.MOUNT_PREFIX;
    if (!prefix) {
      console.error('MOUNT_PREFIX is not configured');
      return new Response('Worker misconfigured: MOUNT_PREFIX missing', { status: 500 });
    }

    const url = new URL(request.url);
    // The frontend loads its assets by relative path, so it must be served
    // from "<prefix>/", never from the bare "<prefix>".
    if (url.pathname === prefix) {
      return Response.redirect(`${url.origin}${prefix}/${url.search}`, 308);
    }
    if (!url.pathname.startsWith(`${prefix}/`)) {
      return new Response('Not Found', { status: 404 });
    }

    const assetUrl = new URL(url);
    assetUrl.pathname = url.pathname.slice(prefix.length);
    const response = await env.ASSETS.fetch(new Request(assetUrl, request));

    const location = response.headers.get('Location');
    if (location && response.status >= 300 && response.status < 400) {
      const target = new URL(location, assetUrl);
      if (target.origin === url.origin && !target.pathname.startsWith(`${prefix}/`)) {
        target.pathname = `${prefix}${target.pathname}`;
        const headers = new Headers(response.headers);
        headers.set('Location', target.toString());
        return new Response(response.body, { status: response.status, headers });
      }
    }
    return response;
  },
};
