# Deploy

Face Cam is static files only, so GitHub Pages is the whole deployment. News
Stack also needs a Cloudflare Worker, because browsers block cross-origin
feeds. Face Cam has no such requests: the model files come from CDNs that send
CORS headers. So there is no Worker.

## 1. Put the app on GitHub Pages

```bash
gh repo create facecam-pwa --public --source=. --push
gh api -X POST repos/joek670/facecam-pwa/pages --field "source[branch]=main" --field "source[path]=/"
```

Or in the browser: **Settings > Pages > Source: Deploy from a branch > main >
/ (root) > Save.**

The URL is `https://joek670.github.io/facecam-pwa/`. The first build takes one
or two minutes.

HTTPS is necessary. Safari does not give camera access, and does not register
a service worker, on a plain-HTTP page.

## 2. Install it on the iPhone

1. Open the URL in Safari.
2. Tap **Share > Add to Home Screen**.
3. Open Face Cam from the Home Screen and tap **Start camera**. Allow the
   camera when Safari asks.

The first start downloads the face model (about 14 MB). After that, the app
works with no connection.

## Updating

Push to `main`. The service worker gets the app files network-first, so the
next launch with a connection uses the new version.

To change the MediaPipe version, change `TASKS_VERSION` in `face.js` and the
model cache name `MODEL` in `sw.js` together. If you change only the version,
the old files stay in the cache with the new ones.
