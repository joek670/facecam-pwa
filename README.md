# Face Cam

A camera app with a reference-photo face filter, built as an installable PWA
like [News Stack](https://github.com/joek670/newsstack-v3-pwa). Choose a photo
of a face, point the camera at a person, and the face from the photo follows
the live face: head turns, jaw, and expression.

Face tracking runs on the phone. No frame, photo, or shot is sent anywhere.

## Features

- Live camera preview, front and back. The front camera view is mirrored.
- Reference photo: choose any photo with a clear face. The app finds the face
  and refuses a photo with no face in it. The photo is kept in this browser
  (IndexedDB), so it is still set after a relaunch.
- Face filter: the reference face is drawn onto the live face, with the edge
  faded into the live skin.
- Blend: from 0% (live face only) to 100% (reference face only).
- Match skin tone: shifts the reference colours toward the live face's
  lighting.
- Tracking mesh: shows the 468 tracked points as a wireframe.
- Capture: saves what is on screen as a JPEG. Tap a shot to share it or save
  it to Photos. Shots are kept only until the tab closes.
- Works offline after the first load. Light, dark, or system theme.

## How the filter works

[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
finds 468 points on a face. Each point has the same meaning in every face:
point 1 is always the nose tip. So the reference photo and the live frame
match point for point, with no alignment step.

The app joins the points into MediaPipe's 854-triangle face mesh. For each
frame, it draws each triangle of the reference photo at the position of the
same triangle on the live face (WebGL, `face.js`). The mesh has no triangles
across the mouth, so an open mouth shows the live mouth.

The model and the WASM runtime come from Google's storage and jsDelivr, pinned
to `@mediapipe/tasks-vision@0.10.14`. The service worker keeps them after the
first load (about 14 MB).

## Files

| File | What it does |
| --- | --- |
| `index.html` | Layout, styles, and the Content Security Policy |
| `app.js` | Camera, reference photo, render loop, capture |
| `face.js` | Face Landmarker set-up, the face mesh, and the WebGL warp |
| `theme.js` | System / Light / Dark control |
| `sw.js` | Offline cache: app files network-first, model files cache-first |
| `manifest.webmanifest`, `icon*` | Home Screen install |

## Run it locally

The camera needs HTTPS or localhost. Any static server works:

```bash
python -m http.server 8031 --bind 127.0.0.1
```

Then open <http://127.0.0.1:8031/>. To use a phone, deploy it (see
[DEPLOY.md](DEPLOY.md)). A phone cannot open the camera from a plain-HTTP LAN
address.

## Use it responsibly

Use a person's face only with that person's permission. Do not use shots to
deceive anyone about who is in them.

## License

MIT. MediaPipe is Apache 2.0 and is loaded at run time, not included here.
