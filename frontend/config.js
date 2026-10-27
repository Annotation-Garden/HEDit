// Backend API Configuration
// The frontend lives at annotation.garden/hedit/ (dev: annotation.garden/hedit-dev/),
// and calls its browser-facing Worker at edge.annotation.garden/hedit (dev: /hedit-dev).
// Neither depends on an account-scoped *.pages.dev or *.workers.dev name.
(function() {
    const { hostname, pathname, search, hash } = window.location;

    // The legacy hedit Pages project still builds hedit.pages.dev (main) and
    // develop.hedit.pages.dev (develop) until it is retired; send those
    // visitors to the same page on annotation.garden. Other *.hedit.pages.dev
    // previews are left alone for pull-request testing. The configuration
    // below still runs: replace() navigates asynchronously, and the page's
    // scripts start calling the backend before it does.
    const legacyMounts = { 'hedit.pages.dev': '/hedit', 'develop.hedit.pages.dev': '/hedit-dev' };
    if (Object.hasOwn(legacyMounts, hostname)) {
        window.location.replace(`https://annotation.garden${legacyMounts[hostname]}${pathname}${search}${hash}`);
    }

    // Development: the /hedit-dev/ path, develop.hedit.pages.dev (until the
    // redirect above completes), and local runs
    const isDev = pathname.startsWith('/hedit-dev/') ||
                  hostname.startsWith('develop.') ||
                  hostname.includes('localhost') ||
                  hostname.includes('127.0.0.1');

    if (isDev) {
        // Dev backend (Cloudflare Worker proxy to dev container)
        window.BACKEND_URL = 'https://edge.annotation.garden/hedit-dev';
        // Turnstile testing key (always passes) - pairs with testing secret key
        window.TURNSTILE_SITE_KEY = '1x00000000000000000000AA';
        console.log('[HEDit] Using DEV backend:', window.BACKEND_URL);
    } else {
        // Production backend (Cloudflare Worker proxy to prod container)
        window.BACKEND_URL = 'https://edge.annotation.garden/hedit';
        // Production Turnstile site key
        window.TURNSTILE_SITE_KEY = '0x4AAAAAACEkzthaT1R2kLIF';
    }
})();
