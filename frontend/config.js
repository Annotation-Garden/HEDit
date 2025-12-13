// Backend API Configuration
// Auto-detect environment based on hostname
(function() {
    const hostname = window.location.hostname;

    // Development environment: develop.hed-bot.pages.dev
    const isDev = hostname.startsWith('develop.') ||
                  hostname.includes('localhost') ||
                  hostname.includes('127.0.0.1');

    if (isDev) {
        // Dev backend (Cloudflare Worker proxy to dev container)
        window.BACKEND_URL = 'https://hed-bot-dev-api.shirazi-10f.workers.dev';
        console.log('[HED-BOT] Using DEV backend:', window.BACKEND_URL);
    } else {
        // Production backend (Cloudflare Worker proxy to prod container)
        window.BACKEND_URL = 'https://hed-bot-api.shirazi-10f.workers.dev';
    }
})();

// Cloudflare Turnstile Configuration
// Site key for bot protection (safe to expose in frontend)
window.TURNSTILE_SITE_KEY = '0x4AAAAAACEkzthaT1R2kLIF';
