const CACHE_NAME = 'keiba-ai-v1';
const URLS_TO_CACHE = ['/', '/keiba/', '/keiba/index.html'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(URLS_TO_CACHE)));
});

self.addEventListener('fetch', e => {
  // Network first, fallback to cache
  e.respondWith(
    fetch(e.request).catch(() => caches.match(e.request))
  );
});
