// Backend API Configuration
// Auto-detect environment based on hostname
(function() {
    const hostname = window.location.hostname;

    // Development environment: develop.hedit.pages.dev
    const isDev = hostname.startsWith('develop.') ||
                  hostname.includes('localhost') ||
                  hostname.includes('127.0.0.1');

    if (isDev) {
        // Dev backend (Cloudflare Worker proxy to dev container)
        window.BACKEND_URL = 'https://hedit-dev-api.shirazi-10f.workers.dev';
        // Turnstile testing key (always passes) - pairs with testing secret key
        window.TURNSTILE_SITE_KEY = '1x00000000000000000000AA';
        console.log('[HEDit] Using DEV backend:', window.BACKEND_URL);
    } else {
        // Production backend (Cloudflare Worker proxy to prod container)
        window.BACKEND_URL = 'https://hedit-api.shirazi-10f.workers.dev';
        // Production Turnstile site key
        window.TURNSTILE_SITE_KEY = '0x4AAAAAACEkzthaT1R2kLIF';
    }
})();
