/* Face Cam service worker.
 *
 * Two caches. The app shell is network-first, as in News Stack, so a redeploy
 * is picked up on the next launch and the cache is the offline fallback.
 *
 * The MediaPipe bundle, WASM and models are cache-first. Their URLs carry a
 * pinned version, so a cached copy is never stale, and they are ~14 MB that a
 * phone should download once, not on every launch. Caching them is what
 * makes the app work with no signal after the first good load. The hair and
 * background model (16 MB more) is fetched only when one of those effects is
 * first used, and cached the same way from then on.
 */
const SHELL = 'facecam-shell-6';
const MODEL = 'facecam-model-1';
const ASSETS = [
  './', './index.html', './app.js', './face.js', './theme.js', './settings.js',
  './interpret.js', './config.js',
  './manifest.webmanifest', './icon.svg', './icon-180.png', './icon-192.png',
  './icon-512.png',
];
// URL prefixes. The second covers both models: face_landmarker/... and
// image_segmenter/selfie_multiclass_256x256/float32/1/...
const PINNED = [
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@',
  'https://storage.googleapis.com/mediapipe-models/',
];

self.addEventListener('install', e =>
  e.waitUntil(
    caches.open(SHELL)
      // One missing icon must not fail the whole install.
      .then(c => Promise.allSettled(ASSETS.map(a => c.add(a))))
      .then(() => self.skipWaiting())
  )
);

self.addEventListener('activate', e =>
  e.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== SHELL && k !== MODEL).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  )
);

self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  const url = e.request.url;

  if (PINNED.some(p => url.startsWith(p))) {
    e.respondWith(
      caches.open(MODEL).then(c =>
        c.match(e.request).then(hit => hit || fetch(e.request).then(res => {
          if (res.ok) c.put(e.request, res.clone()).catch(() => {});
          return res;
        }))
      )
    );
    return;
  }

  if (new URL(url).origin !== self.location.origin) return;
  e.respondWith(
    // no-cache: revalidate with the server every time (a 304 is cheap).
    // GitHub Pages sends max-age=600, and without this a plain fetch can be
    // answered from the browser cache: a new index.html then runs with the
    // old app.js, which finds none of its controls and stops at start-up.
    fetch(e.request, { cache: 'no-cache' })
      .then(res => {
        const copy = res.clone();
        caches.open(SHELL).then(c => c.put(e.request, copy)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(e.request).then(hit => hit || caches.match('./index.html')))
  );
});
