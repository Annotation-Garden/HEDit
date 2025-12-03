// Backend API Configuration
// Cloudflare Worker proxy with caching and rate limiting
// Backend runs via Cloudflare Tunnel (private, not directly exposed)
window.BACKEND_URL = 'https://hed-bot-api.shirazi-10f.workers.dev';

// Cloudflare Turnstile Configuration
// Site key for bot protection (safe to expose in frontend)
window.TURNSTILE_SITE_KEY = '0x4AAAAAACEkzthaT1R2kLIF';
