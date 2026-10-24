# Deploy

Two pieces, like News Stack: the app goes on GitHub Pages, and an optional
Cloudflare Worker reads "Describe a look" with Claude. The app is complete
without the Worker; the text box then uses the on-device word list.

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

## 3. Optional: the Claude interpreter

The Worker keeps the Anthropic API key off the phone. It accepts requests only
from the origins in `ALLOWED_ORIGINS` in `worker/src/index.js`, limits each IP
address to 20 requests a minute (`wrangler.toml`), caps the text at 300
characters, and sends Claude only that text and the current slider values.

Each request is one call to `claude-opus-5-5` at low effort. Server-side
fallbacks are on: if Claude's safety checks decline a request, the API re-runs
it on its recommended fallback model.

From the `worker` folder (PowerShell):

```powershell
npm install
npx wrangler login
npx wrangler secret put ANTHROPIC_API_KEY
npx wrangler deploy
```

`secret put` asks for the key at a hidden prompt. Never put the key in
`wrangler.toml` or in the repository.

`deploy` prints the Worker URL, for example
`https://facecam-interpreter.<account>.workers.dev`. Check it answers:

```powershell
Invoke-RestMethod -Method Post -Uri "https://facecam-interpreter.<account>.workers.dev/interpret" -Headers @{ Origin = "https://joek670.github.io" } -ContentType "application/json" -Body '{"text":"blue eyes, a little warmer"}'
```

Replace `<account>` with your subdomain from the deploy output. A working
Worker returns a `patch` with `eyeColor`, `eyeAmt`, and `warmth`.

Then set the URL in `config.js` and push:

```js
export const INTERPRETER_URL = 'https://facecam-interpreter.<account>.workers.dev';
```

The page's Content Security Policy already allows `https://*.workers.dev`. On
a custom domain, add it to `connect-src` in `index.html`.

## Updating

Push to `main`. The service worker gets the app files network-first, so the
next launch with a connection uses the new version.

To change the MediaPipe version, change `TASKS_VERSION` in `face.js` and the
model cache name `MODEL` in `sw.js` together. If you change only the version,
the old files stay in the cache with the new ones.

Redeploy the Worker with `npx wrangler deploy` from `worker`.
