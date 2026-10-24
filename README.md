# Face Cam

A camera app with face filters, built as an installable PWA like
[News Stack](https://github.com/joek670/newsstack-v3-pwa). Choose a reference
photo and the camera gives your face its look: its makeup, lipstick, blush,
skin tone and eye colour, on your own features. Or describe a look in words:
"blue eyes, soft skin, a little warmer".

Face tracking and every filter run on the phone. No frame, photo, or shot is
sent anywhere. The optional "Describe a look" interpreter receives the typed
text only.

## Features

- **Camera:** live preview, front and back. The front camera view is mirrored.
  Capture saves what is on screen as a JPEG. Tap a shot to share it or save it
  to Photos. Shots are kept only until the tab closes.
- **From the reference photo:** choose the areas (whole face, eyes, brows,
  nose, lips) and how the photo is used:
  - **Look** (the default): the photo works as a filter. Your face takes its
    makeup, lipstick, blush, skin tone and iris colour, and keeps its own
    features, expression and lighting. Nothing of the photo's face is drawn.
  - **Face swap**: your face takes the photo's proportions (eye size and
    spacing, nose, mouth, brows) and its colours at full strength. The
    photo's pixels are never drawn over you: the face on screen is always
    your live face, reshaped and recoloured, so it blinks, talks and keeps
    the camera's lighting.

  Pick an anime or cartoon face and the app sees that its skin is flat
  colour, not photographed skin, and turns on Face swap and Cartoon style.

  The photo is kept in this browser (IndexedDB) and is still set after a
  relaunch.
- **Edits, no reference needed:** eye colour, lip colour, skin smoothing, eye
  size, and a cartoon style for the whole picture. A colour grade (warmth, brightness, contrast, saturation) applies
  to the whole image.
- **Describe a look:** type what you want; the app sets the controls and says
  what it understood and what it did not. The sliders stay the source of
  truth, so you can adjust after.
- The camera stays in view while you scroll through the settings: at the top
  on a phone, on the left on a wide screen.
- Works offline after the first load. Light, dark, or system theme.

## How the filter works

[MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
finds 478 points on a face, 10 of them on the irises. Each point has the same
meaning in every face: point 1 is always the nose tip. So the reference photo
and the live frame match point for point, with no alignment step.

`face.js` joins the points into MediaPipe's face mesh (852 triangles, plus 28
that fill the eye openings, which MediaPipe leaves open) and redraws the live
face through it each frame in WebGL. One pass does all the face work:

- **Feature masks:** each feature's outline is drawn into a mask and blurred,
  so the photo changes only the chosen areas and fades out at their edges.
- **Look:** the photo and the camera image are both blurred to a scale where
  makeup, lipstick, blush and skin tone survive but eyes, pores and feature
  shapes do not. The live face is recoloured by the ratio of the two, so it
  keeps all its own detail. Darkening transfers at half strength, measured
  against each face's average, so a darker photo does not darken you. The
  eye openings are left out, and the irises take the photo's measured iris
  colour instead.
- **Face swap:** each live point moves by the difference between where the
  photo has it and where your face has it at rest, both measured in the
  face's own frame (origin between the eyes, one unit the eye spacing). The
  rest shape is a slow average of the live face, so a blink or a smile still
  moves the face. The face outline stays put, so the background never shows
  a second jaw. The face is then recoloured as in Look, at full strength.
- **Cartoon style:** the frame is blurred twice. The smaller blur is the flat
  colour, shading is cut into four soft bands, and where the smaller blur is
  darker than the larger one there is an edge, drawn as an ink line.
- **Drawn-face check:** the median Laplacian of the skin on the cheeks and
  forehead. Photos measured 1.7 to 2.5, a photo smoothed at full strength
  0.53, cartoon output 0.24 to 0.28; below 0.4 counts as drawn.
- **Eye colour:** a ring between the pupil and the iris edge, inside the eye
  opening, so the pupil and the whites keep their colour.
- **Eye size:** the mesh points around each iris move outward, most at the
  centre and not at all 1.5 eye widths away.
- **Skin smoothing:** edge-aware; it softens skin but leaves edges such as the
  jaw and nostrils sharp, and it skips the eyes, brows, and mouth.

The model and the WASM runtime come from Google's storage and jsDelivr, pinned
to `@mediapipe/tasks-vision@0.10.14`. The service worker keeps them after the
first load (about 14 MB).

## Describe a look

With no interpreter configured, `interpret.js` reads the text on the device
with a fixed word list: colours, "from my photo", "slightly", "very",
percentages, "black and white", "no lipstick", and similar.

With the Worker in `worker/` deployed (see [DEPLOY.md](DEPLOY.md), step 3),
Claude reads the text and fills in the settings schema, which handles loose
phrasing ("make my eyes pop", "softer than now"). The Worker sends Claude the
typed text and the current slider values, nothing else. If the Worker cannot
be reached, the app falls back to the word list and says so.

Both paths go through `clean()` in `settings.js`, which drops anything outside
the schema.

## Files

| File | What it does |
| --- | --- |
| `index.html` | Layout, styles, and the Content Security Policy |
| `app.js` | Camera, reference photo, controls, render loop, capture |
| `face.js` | Face Landmarker set-up, the mesh, feature masks, and the WebGL renderer |
| `settings.js` | Settings defaults, limits, validation, and read-back text |
| `interpret.js` | "Describe a look": the on-device word list and the Worker call |
| `config.js` | The interpreter URL (empty: on-device only) |
| `theme.js` | System / Light / Dark control |
| `sw.js` | Offline cache: app files network-first, model files cache-first |
| `manifest.webmanifest`, `icon*` | Home Screen install |
| `worker/` | Optional Cloudflare Worker that reads the text with Claude |

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
