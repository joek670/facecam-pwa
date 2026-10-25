/* Face Cam — face tracking, the reference-photo transfer, and the edits.
 *
 * MediaPipe Face Landmarker gives 478 points per face: 468 mesh points plus
 * 10 iris points. The mesh points have the same meaning in every image —
 * point 1 is always the nose tip — so a reference photo and a live frame can
 * be matched point for point with no alignment step.
 *
 * Every frame, the live face is redrawn through the mesh. That one pass does
 * all the face work:
 *   - the reference photo, in one of two modes, limited by a soft mask to
 *     the chosen areas (whole face, eyes, brows, nose, lips):
 *       Look (the default) recolours the live face with the photo's colours:
 *       makeup, lipstick, blush, skin tone, iris colour. The live face keeps
 *       its own features; nothing of the photo's face is drawn.
 *       Features (a face swap) also moves the live face's points to the
 *       photo's proportions, then recolours it with the photo's colours at
 *       full strength. The photo's pixels are never drawn: the face is
 *       always the live one, so it blinks, talks and keeps its lighting.
 *   - edits with no reference: skin smoothing, lip colour, eye colour, and
 *     eye size (a warp of the mesh points around each eye).
 *   - a cartoon style (flat colour, banded shading, ink lines) over the
 *     whole frame, for a drawn or anime reference.
 *   - filters that need no photo (anime; a gender swap either way): a
 *     smooth reshaping of the mesh in the face's own frame, and colour
 *     worked out on the live face (blush, lip colour, stubble).
 *   - a colour grade (warmth, brightness, contrast, saturation), applied to
 *     the face and the background alike so there is no seam.
 *
 * When a filter moves the face outline, the background is drawn through a
 * grid around the face that moves with it, so the real jaw never shows
 * beside the new one.
 *
 * A second model, MediaPipe Image Segmenter, sorts every pixel of the frame
 * into hair, background, skin, clothes and others. The background pass uses
 * it to recolour hair and to blur or replace the background, and the face
 * pass recolours the hair it covers (a fringe) the same way.
 *
 * Everything runs on the device. The models and WASM come from the CDN once and
 * the service worker keeps them; no frame or photo leaves the browser.
 */

export const TASKS_VERSION = '0.10.14';
const TASKS_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const SEG_URL =
  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite';
// The multiclass model's classes, in this order (its model card):
// background, hair, body skin, face skin, clothes, others.
const SEG_BG = 0, SEG_HAIR = 1;
// The segmenter sees 256 x 256 whatever it is given, so the camera frame is
// scaled to this on its long side first. The masks come back the same size,
// and reading two of them back from the GPU each frame is then cheap.
const SEG_SIZE = 320;
// The masks' feather, in mask pixels: enough that no effect has a hard edge,
// little enough that the edge does not spread into a rim.
const SEG_FEATHER = 0.8;
// The darkest a recoloured hair pixel gets (see tint() in TINT). Higher than
// lips: most hair is darker than lips, and at 0.35 dark hair barely changes.
const HAIR_LIFT = 0.45;

// The 468 mesh points. The iris points (468-477) are not part of the tesselation.
const MESH_POINTS = 468;
const ALL_POINTS = 478;

// MediaPipe publishes no nose contour. These points outline the nose from the
// bridge between the eyes, down one side, round the nostrils and back up.
// Checked against a detected face: the pairs (193/417, 245/465, ...) mirror.
const NOSE = [168, 193, 245, 188, 174, 236, 198, 209, 129, 64, 98, 97, 2,
  326, 327, 294, 358, 429, 420, 456, 399, 412, 465, 417];

// Face width is measured cheek to cheek; every size below is a fraction of it,
// so the filter behaves the same near and far from the camera.
const CHEEK_L = 234, CHEEK_R = 454;

export const REGIONS = ['face', 'eyes', 'brows', 'nose', 'lips'];

// Filters with no reference photo. Each reshapes and recolours the live face;
// none draws anything that is not the camera image.
export const FILTERS = ['none', 'anime', 'feminine', 'masculine'];

// The most the face outline may move, in face widths, and the reach of the
// background's movement around it (a Gaussian's sigma). The background grid
// stretches by at most about EDGE_MAX / BG_SIGMA * 0.6, well under 1, so it
// never folds.
const EDGE_MAX = 0.1;
const BG_SIGMA = 0.13;
// Grid cells on each side of the background grid around the face.
const GRID = 40;

let vision = null;
let FaceLandmarker = null;
let ImageSegmenter = null;
let shapes = null;

async function loadTasks() {
  if (vision) return;
  const mod = await import(`${TASKS_BASE}/vision_bundle.mjs`);
  FaceLandmarker = mod.FaceLandmarker;
  ImageSegmenter = mod.ImageSegmenter;
  vision = await mod.FilesetResolver.forVisionTasks(`${TASKS_BASE}/wasm`);
}

/** A landmarker in 'IMAGE' or 'VIDEO' mode. GPU first, CPU if the GPU path fails. */
export async function createLandmarker(runningMode) {
  await loadTasks();
  const opts = delegate => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode,
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await FaceLandmarker.createFromOptions(vision, opts('GPU'));
  } catch (err) {
    console.warn('GPU delegate failed, using CPU', err);
    return FaceLandmarker.createFromOptions(vision, opts('CPU'));
  }
}

/**
 * A hair and background segmenter in 'IMAGE' or 'VIDEO' mode, GPU first.
 * Confidence masks, not the category mask: a confidence is soft at the edge
 * of a class (stray hairs are half hair), where a category is all or nothing.
 */
export async function createSegmenter(runningMode) {
  await loadTasks();
  const opts = delegate => ({
    baseOptions: { modelAssetPath: SEG_URL, delegate },
    runningMode,
    outputConfidenceMasks: true,
    outputCategoryMask: false,
  });
  try {
    return await ImageSegmenter.createFromOptions(vision, opts('GPU'));
  } catch (err) {
    console.warn('GPU delegate failed, using CPU', err);
    return ImageSegmenter.createFromOptions(vision, opts('CPU'));
  }
}

// A confidence as a mask. The model rarely goes above 0.9 even where it is
// sure (the rest is spread over the other five classes), and at 0.9 a
// replaced background would still show a tenth of the real one. So lo + 0.5
// and up counts as all, lo and down as none, with a smooth step between.
function sure(c, lo) {
  const t = Math.min(1, Math.max(0, (c - lo) * 2));
  return t * t * (3 - 2 * t);
}
// The background's step starts lower, which moves its edge a pixel or so
// into the person. Edge pixels are part real background; given to the
// person, they would show it as a coloured rim around a replaced one.
const SURE_HAIR = 0.25, SURE_BG = 0.1;

/**
 * Hair and background of one camera frame, as RGBA bytes: hair in red,
 * background in green. `out` is reused when it is the right size.
 * Returns { data, w, h }, in camera-image orientation like the video texture.
 */
export function segmentFrame(segmenter, source, sw, sh, canvas, out) {
  const k = SEG_SIZE / Math.max(sw, sh);
  const cw = Math.max(1, Math.round(sw * k)), ch = Math.max(1, Math.round(sh * k));
  if (canvas.width !== cw) canvas.width = cw;
  if (canvas.height !== ch) canvas.height = ch;
  canvas.getContext('2d').drawImage(source, 0, 0, cw, ch);
  let frame = null;
  // The masks are valid only inside the callback, so they are read there.
  segmenter.segmentForVideo(canvas, performance.now(), r => {
    const bgMask = r.confidenceMasks[SEG_BG], hairMask = r.confidenceMasks[SEG_HAIR];
    const w = bgMask.width, h = bgMask.height;
    const bg = bgMask.getAsFloat32Array(), hair = hairMask.getAsFloat32Array();
    const data = out?.length === w * h * 4 ? out : new Uint8ClampedArray(w * h * 4);
    for (let i = 0, j = 0; i < w * h; i++, j += 4) {
      data[j] = sure(hair[i], SURE_HAIR) * 255;
      data[j + 1] = sure(bg[i], SURE_BG) * 255;
      data[j + 3] = 255;
    }
    frame = { data, w, h };
  });
  return frame;
}

/**
 * The reference photo's hair colour, as the tint that gives hair as bright
 * as the photo's exactly that colour (see tint() in TINT), or null when the
 * photo shows too little hair. Sampled once per photo, with a segmenter of
 * its own in IMAGE mode that is closed after: the live one is in VIDEO mode.
 */
export async function referenceHair(image) {
  const seg = await createSegmenter('IMAGE');
  try {
    const c = document.createElement('canvas');
    c.width = image.width;
    c.height = image.height;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(image, 0, 0);
    const px = ctx.getImageData(0, 0, c.width, c.height).data;
    let r = 0, g = 0, b = 0, n = 0;
    seg.segment(image, res => {
      const m = res.confidenceMasks[SEG_HAIR];
      const conf = m.getAsFloat32Array();
      for (let y = 0; y < m.height; y++) {
        const iy = Math.min(c.height - 1, Math.floor((y + 0.5) * c.height / m.height));
        for (let x = 0; x < m.width; x++) {
          // Only pixels that are surely hair: the edges are part background.
          const w = conf[y * m.width + x];
          if (w < 0.7) continue;
          const k = (iy * c.width + Math.min(c.width - 1, Math.floor((x + 0.5) * c.width / m.width))) * 4;
          r += px[k] * w; g += px[k + 1] * w; b += px[k + 2] * w; n += w;
        }
      }
    });
    if (n < 0.002 * c.width * c.height) return null;
    const mean = [r / n / 255, g / n / 255, b / n / 255];
    const lum = 0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2];
    return mean.map(v => Math.min(1, v / (HAIR_LIFT + (1.75 - HAIR_LIFT) * lum)));
  } finally {
    seg.close();
  }
}

/** The mesh as triangles and line pairs, built once from MediaPipe's edge list. */
export function buildMesh() {
  const edges = FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  const adj = Array.from({ length: MESH_POINTS }, () => new Set());
  for (const { start, end } of edges) {
    adj[start].add(end);
    adj[end].add(start);
  }
  // The tesselation lists edges only, so triangles are found as 3-cycles of
  // edges. That finds 854. Two of them are not faces: at each nostril
  // (49/64/129, 279/294/358) three edges enclose another point, and every
  // edge of such a cycle is shared by three triangles instead of two. Without
  // them there are 852, and 1322 edges - 468 points + 1 = 855 faces leaves
  // three holes: the mouth and both eye openings.
  const cycles = [];
  for (let a = 0; a < MESH_POINTS; a++) {
    for (const b of adj[a]) {
      if (b <= a) continue;
      for (const c of adj[b]) {
        if (c <= b || !adj[a].has(c)) continue;
        cycles.push([a, b, c]);
      }
    }
  }
  const key = (a, b) => (a < b ? a * 1000 + b : b * 1000 + a);
  const uses = new Map();
  const sides = ([a, b, c]) => [key(a, b), key(b, c), key(c, a)];
  for (const t of cycles) for (const k of sides(t)) uses.set(k, (uses.get(k) || 0) + 1);
  const tris = [];
  for (const t of cycles) if (!sides(t).every(k => uses.get(k) > 2)) tris.push(...t);

  // Fill the eye openings, a fan from one corner of each eyelid outline.
  // Without them nothing drawn through the mesh reaches the eyeball: no eye
  // colour, no eye transfer, and the eye-size warp would move only the lids.
  // The mouth stays open, so an open live mouth shows the live teeth.
  for (const loop of [...chains(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE), ...chains(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE)]) {
    for (let i = 1; i < loop.length - 1; i++) tris.push(loop[0], loop[i], loop[i + 1]);
  }
  const lines = [];
  for (const { start, end } of edges) lines.push(start, end);

  // Feather the outline: opacity rises over the three rings in from the face
  // edge, so nothing the face pass draws can leave a seam with the background.
  const ring = new Int16Array(MESH_POINTS).fill(-1);
  let frontier = [...new Set(FaceLandmarker.FACE_LANDMARKS_FACE_OVAL.flatMap(e => [e.start, e.end]))];
  frontier.forEach(i => (ring[i] = 0));
  for (let r = 1; r <= 3 && frontier.length; r++) {
    const next = [];
    for (const i of frontier) for (const j of adj[i]) if (ring[j] < 0) { ring[j] = r; next.push(j); }
    frontier = next;
  }
  const FEATHER = [0, 0.45, 0.8, 1];
  const alpha = new Float32Array(MESH_POINTS);
  for (let i = 0; i < MESH_POINTS; i++) alpha[i] = ring[i] < 0 ? 1 : FEATHER[ring[i]];

  // Face swap shape weights: how far each point moves toward the reference's
  // proportions, per area, fading over two rings around the area. The face
  // outline does not move. With the background following it (see
  // Renderer.warpBackground) it could, but measured against a known target
  // that was no closer on the outline and a little further off elsewhere.
  const grow = (seed, steps) => {
    const w = new Float32Array(ALL_POINTS);
    let f = [...seed];
    f.forEach(i => (w[i] = 1));
    for (const step of steps) {
      const next = [];
      for (const i of f) if (i < MESH_POINTS) for (const j of adj[i]) if (!w[j]) { w[j] = step; next.push(j); }
      f = next;
    }
    return w;
  };
  const ends = edgeList => [...new Set(edgeList.flatMap(e => [e.start, e.end]))];
  const IRIS = Array.from({ length: ALL_POINTS - MESH_POINTS }, (_, i) => MESH_POINTS + i);
  const face = new Float32Array(ALL_POINTS).fill(1);
  face.set(alpha);
  const shape = {
    face,
    eyes: grow([...ends(FaceLandmarker.FACE_LANDMARKS_LEFT_EYE), ...ends(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE), ...IRIS], [0.6, 0.25]),
    brows: grow([...ends(FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW), ...ends(FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW)], [0.5, 0.2]),
    nose: grow([...NOSE, 1, 2, 4, 5, 6, 19, 94, 195, 197], [0.6, 0.25]),
    lips: grow(ends(FaceLandmarker.FACE_LANDMARKS_LIPS), [0.6, 0.25]),
  };

  // The outline, which the background follows, and the outline with the
  // three rings in from it, whose movement is capped (Renderer.capEdge).
  const oval = [], edge = [];
  for (let i = 0; i < MESH_POINTS; i++) {
    if (ring[i] === 0) oval.push(i);
    if (ring[i] >= 0) edge.push(i);
  }

  return { tris: new Uint16Array(tris), lines: new Uint16Array(lines), alpha, shape,
    oval: new Uint16Array(oval), edge: new Uint16Array(edge) };
}

/** Mean RGB of `source` sampled at the mesh points, on a small working canvas. */
export function meanColor(source, points, sw, sh, scratch) {
  const w = 160;
  const h = Math.max(1, Math.round((w * sh) / sw));
  scratch.width = w;
  scratch.height = h;
  const ctx = scratch.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < MESH_POINTS; i++) {
    const x = Math.min(w - 1, Math.max(0, Math.round(points[i].x * w)));
    const y = Math.min(h - 1, Math.max(0, Math.round(points[i].y * h)));
    const k = (y * w + x) * 4;
    // Skip near-black samples: eye corners and nostrils, not skin.
    if (px[k] + px[k + 1] + px[k + 2] < 60) continue;
    r += px[k]; g += px[k + 1]; b += px[k + 2]; n++;
  }
  return n ? [r / n, g / n, b / n] : null;
}

// ---------------------------------------------------------------------------
// Feature outlines and masks
// ---------------------------------------------------------------------------

/** Walk an edge list into point chains: closed loops, or open arcs. */
function chains(edges) {
  const adj = new Map();
  const add = (a, b) => { if (!adj.has(a)) adj.set(a, []); adj.get(a).push(b); };
  for (const { start, end } of edges) { add(start, end); add(end, start); }
  const seen = new Set();
  const out = [];
  // Open arcs start at a point with one neighbour; take those first.
  const order = [...adj.keys()].sort((a, b) => adj.get(a).length - adj.get(b).length);
  for (const s of order) {
    if (seen.has(s)) continue;
    const chain = [s];
    seen.add(s);
    for (let cur = s; ;) {
      const next = adj.get(cur).find(n => !seen.has(n));
      if (next === undefined) break;
      chain.push(next);
      seen.add(next);
      cur = next;
    }
    out.push(chain);
  }
  return out;
}

function buildShapes() {
  const F = FaceLandmarker;
  return {
    eyes: [...chains(F.FACE_LANDMARKS_LEFT_EYE), ...chains(F.FACE_LANDMARKS_RIGHT_EYE)],
    // Each eyebrow is two open arcs, upper and lower; they are joined into
    // one outline at draw time, when the point positions are known.
    brows: [chains(F.FACE_LANDMARKS_LEFT_EYEBROW), chains(F.FACE_LANDMARKS_RIGHT_EYEBROW)],
    lips: chains(F.FACE_LANDMARKS_LIPS),
    oval: chains(F.FACE_LANDMARKS_FACE_OVAL)[0],
    nose: NOSE,
  };
}

function poly(ctx, idx, P) {
  idx.forEach((i, n) => { const [x, y] = P(i); n ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.closePath();
}

/** Upper arc then lower arc, the lower reversed if that closes the outline shorter. */
function joinArcs([a, b], P) {
  if (!b) return a;
  const d = (i, j) => { const [x1, y1] = P(i); const [x2, y2] = P(j); return Math.hypot(x1 - x2, y1 - y2); };
  const tail = a[a.length - 1];
  return d(tail, b[b.length - 1]) < d(tail, b[0]) ? a.concat([...b].reverse()) : a.concat(b);
}

function trace(ctx, name, P) {
  ctx.beginPath();
  if (name === 'eyes') shapes.eyes.forEach(l => poly(ctx, l, P));
  else if (name === 'brows') shapes.brows.forEach(pair => poly(ctx, joinArcs(pair, P), P));
  else if (name === 'lips') shapes.lips.forEach(l => poly(ctx, l, P));
  else if (name === 'nose') poly(ctx, shapes.nose, P);
  else if (name === 'face') poly(ctx, shapes.oval, P);
}

/** Fill a feature, and stroke its outline `grow` wide to take in its surroundings. */
function paint(ctx, name, P, color, grow, rule = 'nonzero') {
  ctx.fillStyle = ctx.strokeStyle = color;
  trace(ctx, name, P);
  ctx.fill(rule);
  if (grow > 0) { ctx.lineWidth = grow; ctx.stroke(); }
}

// How far each transferred feature reaches past its outline, in face widths.
// Eyes take in the lids and lashes; brows the hair either side of the arcs.
const GROW = { eyes: 0.11, brows: 0.07, nose: 0.03, lips: 0.035 };

function prepare(canvas, w, h) {
  if (canvas.width !== w) canvas.width = w;
  if (canvas.height !== h) canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.globalCompositeOperation = 'source-over';
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  return ctx;
}

/** White where the reference replaces the live face. Soft edges come from a GPU blur. */
function drawTransferMask(canvas, w, h, P, fw, regions) {
  const ctx = prepare(canvas, w, h);
  if (regions.includes('face')) {
    paint(ctx, 'face', P, '#fff', 0);
    // Erode the outline so the blur fades out inside the face, not on its edge.
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 0.14 * fw;
    trace(ctx, 'face', P);
    ctx.stroke();
  }
  for (const r of ['eyes', 'brows', 'nose', 'lips']) {
    if (regions.includes(r)) paint(ctx, r, P, '#fff', GROW[r] * fw, r === 'lips' ? 'evenodd' : 'nonzero');
  }
}

/** White on skin only: the face less eyes, brows and mouth. Smoothing uses it. */
function drawSkinMask(canvas, w, h, P, fw) {
  const ctx = prepare(canvas, w, h);
  paint(ctx, 'face', P, '#fff', 0);
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 0.1 * fw;
  trace(ctx, 'face', P);
  ctx.stroke();
  paint(ctx, 'eyes', P, '#000', 0.1 * fw);
  paint(ctx, 'brows', P, '#000', 0.05 * fw);
  paint(ctx, 'lips', P, '#000', 0.04 * fw);
}

/**
 * Hard-edged: red on the lips, green in the eye openings. Recolouring uses it.
 * With browFw (the face width at this size), the brows too, in blue, grown and
 * with a soft edge from three strokes laid over each other. With lashFw
 * instead, a lash line in blue along each upper lid, with a small wing past
 * the outer corner.
 */
function drawFeatureMask(canvas, w, h, P, browFw = 0, lashFw = 0) {
  const ctx = prepare(canvas, w, h);
  paint(ctx, 'lips', P, '#f00', 0, 'evenodd');
  paint(ctx, 'eyes', P, '#0f0', 0);
  if (browFw > 0) {
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 1 / 3;
    for (const g of [0.012, 0.03, 0.05]) paint(ctx, 'brows', P, '#00f', g * browFw);
    ctx.globalAlpha = 1;
  }
  if (lashFw > 0) {
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 1 / 3;
    ctx.strokeStyle = '#00f';
    ctx.lineCap = ctx.lineJoin = 'round';
    for (const g of [0.008, 0.016, 0.026]) {
      ctx.lineWidth = g * lashFw;
      for (const lid of LASH_LINES) {
        // The wing: on along the eye's axis from the outer corner, and up,
        // away from the line between the corners toward the lid's top.
        const [ox, oy] = P(lid[0]), [ix, iy] = P(lid[lid.length - 1]), [tx, ty] = P(lid[4]);
        const ux = tx - (ox + ix) / 2, uy = ty - (oy + iy) / 2;
        ctx.beginPath();
        ctx.moveTo(ox + 0.22 * (ox - ix) + 0.5 * ux, oy + 0.22 * (oy - iy) + 0.5 * uy);
        for (const i of lid) ctx.lineTo(...P(i));
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
  }
}

/** Each upper lid from the outer corner to the inner one. */
const LASH_LINES = [
  [33, 246, 161, 160, 159, 158, 157, 173, 133],
  [263, 466, 388, 387, 386, 385, 384, 398, 362],
];

/**
 * Eye size: push points away from each iris centre, most near the centre and
 * not at all at 1.5 eye widths, so the rest of the face is untouched. The
 * push grows smoothly from zero, so the mesh never folds over itself.
 */
function displace(out, amount, W, H) {
  if (amount <= 0) return out;
  const k = amount * 0.3;
  for (const [c, a, b] of [[468, 33, 133], [473, 362, 263]]) {
    const cx = out[c * 2] * W, cy = out[c * 2 + 1] * H;
    const R = 1.5 * Math.hypot((out[a * 2] - out[b * 2]) * W, (out[a * 2 + 1] - out[b * 2 + 1]) * H);
    for (let i = 0; i < ALL_POINTS; i++) {
      const dx = out[i * 2] * W - cx, dy = out[i * 2 + 1] * H - cy;
      const d = Math.hypot(dx, dy);
      if (d >= R) continue;
      const u = 1 - (d / R) ** 2;
      const f = 1 + k * u * u;
      out[i * 2] = (cx + dx * f) / W;
      out[i * 2 + 1] = (cy + dy * f) / H;
    }
  }
  return out;
}

// The filters' shapes, in the face's own frame (one unit the eye spacing, y
// down). eyeX/eyeY, nose and mouthX/mouthY grow (+) or shrink (-) a feature
// about its centre; jaw draws the lower face in (-) or out (+) toward the
// midline; chin shortens (-) or lengthens (+) the face below the nose; brow
// moves the brows up (-) or down (+); cheek fills the cheekbones out (+);
// size grows (+) or shrinks (-) the whole face about the point between the
// eyes. Sizes from facial dimorphism studies: men have wider jaws, longer
// lower faces, lower brows and faces some 6% larger; women larger eyes,
// fuller lips, smaller noses and fuller cheekbones over a narrower jaw. The
// gender swap doubles the typical difference or so, to read at a glance.
const SHAPES = {
  anime: { eyeX: 0.28, eyeY: 0.5, nose: -0.45, mouthX: -0.4, mouthY: -0.25, jaw: -0.16, chin: -0.12, brow: -0.05, cheek: 0, size: 0 },
  feminine: { eyeX: 0.08, eyeY: 0.12, nose: -0.14, mouthX: -0.04, mouthY: 0.24, jaw: -0.13, chin: -0.07, brow: -0.06, cheek: 0.07, size: -0.04 },
  masculine: { eyeX: -0.05, eyeY: -0.08, nose: 0.08, mouthX: 0.05, mouthY: -0.2, jaw: 0.12, chin: 0.07, brow: 0.05, cheek: 0, size: 0.05 },
};

/**
 * A filter's shape: move each point by a smooth field of where it is on the
 * live face. Each part fades to nothing a little past its feature, and each
 * is gentle enough that the mesh never folds. A blink or a smile still moves
 * the face: the field acts on the live points, not on a fixed shape.
 * L is the live points in the face's frame `fr`; d is moved in place.
 */
function reshape(d, L, fr, browW, k, amt, W, H) {
  const at = i => [L[i * 2], L[i * 2 + 1]];
  const mid = (a, b) => [(L[a * 2] + L[b * 2]) / 2, (L[a * 2 + 1] + L[b * 2 + 1]) / 2];
  const dist = (a, b) => Math.hypot(L[a * 2] - L[b * 2], L[a * 2 + 1] - L[b * 2 + 1]);
  // 1 at the centre, 0 at radius 1, as in displace(): never folds for |k| < 1.
  const bump = u2 => (u2 < 1 ? (1 - u2) ** 2 : 0);
  const smooth = t => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
  const eyes = [[33, 133], [362, 263]].map(([a, b]) => [...mid(a, b), 1.4 * dist(a, b)]);
  const [mx, my] = mid(61, 291);
  const mr = 0.8 * dist(61, 291);
  const tip = at(1), top = at(168);
  const nx = 0.35 * top[0] + 0.65 * tip[0], ny = 0.35 * top[1] + 0.65 * tip[1];
  // Wide enough to take in the nostril wings (about 0.4 either side).
  const nrx = 0.75, nry = 0.6 * Math.abs(tip[1] - top[1]) + 0.1;
  const [chinX, chinY] = at(152);
  const { ux, uy, s } = fr;
  for (let i = 0; i < ALL_POINTS; i++) {
    const x = L[i * 2], y = L[i * 2 + 1];
    let dx = 0, dy = 0;
    for (const [ex, ey, R] of eyes) {
      const b = bump(((x - ex) ** 2 + (y - ey) ** 2) / (R * R));
      dx += k.eyeX * b * (x - ex);
      dy += k.eyeY * b * (y - ey);
    }
    let b = bump(((x - nx) / nrx) ** 2 + ((y - ny) / nry) ** 2);
    dx += k.nose * b * (x - nx);
    dy += k.nose * b * (y - ny);
    b = bump(((x - mx) / mr) ** 2 + ((y - my) / (0.8 * mr)) ** 2);
    dx += k.mouthX * b * (x - mx);
    dy += k.mouthY * b * (y - my);
    // The jaw, from the cheekbones down, toward the line from between the
    // eyes to the chin. Nothing above the cheekbones moves.
    const t = smooth((y - 0.2 * chinY) / (0.6 * chinY));
    dx += k.jaw * (x - chinX * Math.min(1, Math.max(0, y / chinY))) * t * t;
    // The lower face, from just above the nose tip: a ramp with no kink.
    const u = y - tip[1] + 0.25;
    if (u > 0) dy += k.chin * u * u / (u + 0.25);
    dy += k.brow * browW[i];
    // The cheekbones, out to the side, below and outside each eye.
    dx += k.cheek * Math.sign(x) * bump(((Math.abs(x) - 0.95) ** 2 + (y - 0.45) ** 2) / 0.3);
    dx += k.size * x;
    dy += k.size * y;
    dx *= amt;
    dy *= amt;
    d[i * 2] += s * (dx * ux - dy * uy) / W;
    d[i * 2 + 1] += s * (dx * uy + dy * ux) / H;
  }
  return d;
}

/**
 * Points in the face's own frame: origin between the eyes, x along the line
 * from one eye to the other, one unit the distance between them. The same
 * face gives the same local points wherever it is in the image, at any
 * size or roll, so a photo's proportions can be laid onto the live face.
 * `pts` is flat [x0, y0, x1, y1, ...] in 0..1 image coordinates.
 */
function faceFrame(pts, W, H) {
  const mid = (a, b) => [(pts[a * 2] + pts[b * 2]) / 2 * W, (pts[a * 2 + 1] + pts[b * 2 + 1]) / 2 * H];
  const [lx, ly] = mid(33, 133);
  const [rx, ry] = mid(362, 263);
  const s = Math.max(1e-6, Math.hypot(rx - lx, ry - ly));
  return { ox: (lx + rx) / 2, oy: (ly + ry) / 2, ux: (rx - lx) / s, uy: (ry - ly) / s, s };
}

function toLocal(pts, W, H, out) {
  const { ox, oy, ux, uy, s } = faceFrame(pts, W, H);
  for (let i = 0; i < ALL_POINTS; i++) {
    const dx = pts[i * 2] * W - ox, dy = pts[i * 2 + 1] * H - oy;
    out[i * 2] = (dx * ux + dy * uy) / s;
    out[i * 2 + 1] = (dy * ux - dx * uy) / s;
  }
  return out;
}

function flatten(points, out) {
  for (let i = 0; i < ALL_POINTS; i++) { out[i * 2] = points[i].x; out[i * 2 + 1] = points[i].y; }
  return out;
}

/** Iris centre and radius in pixels, from the five iris points starting at `c`. */
function iris(pts, c, W, H) {
  const cx = pts[c * 2] * W, cy = pts[c * 2 + 1] * H;
  let r = 0;
  for (let i = 1; i <= 4; i++) r += Math.hypot(pts[(c + i) * 2] * W - cx, pts[(c + i) * 2 + 1] * H - cy);
  return [cx, cy, r / 4];
}

// Look mode's blur, in face widths: wide enough to remove eyes, pores and
// feature shapes, narrow enough to keep eyeliner, lipstick edges and blush.
const LOOK_BLUR = 1 / 40;

/** Mean colour of both irises in an image, rgb 0..1; the pupil and rim left out. */
function irisColor(image, points) {
  const c = document.createElement('canvas');
  c.width = image.width;
  c.height = image.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  let r = 0, g = 0, b = 0, n = 0;
  for (const k of [468, 473]) {
    const cx = points[k].x * c.width, cy = points[k].y * c.height;
    let rad = 0;
    for (let i = 1; i <= 4; i++) rad += Math.hypot((points[k + i].x - points[k].x) * c.width, (points[k + i].y - points[k].y) * c.height);
    rad /= 4;
    const x0 = Math.max(0, Math.floor(cx - rad)), y0 = Math.max(0, Math.floor(cy - rad));
    const size = Math.max(1, Math.ceil(rad * 2));
    const px = ctx.getImageData(x0, y0, size, size).data;
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const d = Math.hypot(x0 + x - cx, y0 + y - cy) / Math.max(rad, 1);
        if (d < 0.45 || d > 0.85) continue;
        const i = (y * size + x) * 4;
        r += px[i]; g += px[i + 1]; b += px[i + 2]; n++;
      }
    }
  }
  return n ? [r / n / 255, g / n / 255, b / n / 255] : [0.3, 0.2, 0.15];
}

/**
 * The skin texture of the face in `image`: the median Laplacian of
 * brightness, 0..255 scale, on both cheeks and the forehead, with the face
 * scaled to 160 px wide. A photo's skin has grain and pores; anime and
 * cartoon skin is flat fill or a smooth gradient, where the Laplacian is 0.
 * Measured: photos 1.7 to 2.5, a photo smoothed at full strength 0.53, the
 * cartoon style's own output 0.24 to 0.28, flat fill 0 to 0.24.
 */
export function skinTexture(image, points) {
  const W = image.width, H = image.height;
  const scale = 160 / Math.max(faceWidth(points, W, H), 1);
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(W * scale));
  c.height = Math.max(1, Math.round(H * scale));
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, c.width, c.height);
  const px = ctx.getImageData(0, 0, c.width, c.height).data;
  const L = (x, y) => { const k = (y * c.width + x) * 4; return 0.299 * px[k] + 0.587 * px[k + 1] + 0.114 * px[k + 2]; };
  const laps = [];
  // Both cheeks below the eyes, and the forehead above the brows.
  for (const i of [205, 425, 151]) {
    const cx = Math.round(points[i].x * c.width), cy = Math.round(points[i].y * c.height);
    for (let y = cy - 16; y <= cy + 16; y++) {
      for (let x = cx - 16; x <= cx + 16; x++) {
        if (x < 1 || y < 1 || x >= c.width - 1 || y >= c.height - 1) continue;
        laps.push(Math.abs(4 * L(x, y) - L(x - 1, y) - L(x + 1, y) - L(x, y - 1) - L(x, y + 1)) / 4);
      }
    }
  }
  if (!laps.length) return Infinity;
  laps.sort((a, b) => a - b);
  return laps[laps.length >> 1];
}

function faceWidth(points, W, H) {
  return Math.hypot((points[CHEEK_L].x - points[CHEEK_R].x) * W, (points[CHEEK_L].y - points[CHEEK_R].y) * H);
}

// ---------------------------------------------------------------------------
// WebGL renderer
// ---------------------------------------------------------------------------

const PRECISION = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
const vec3 LUMA = vec3(0.299, 0.587, 0.114);`;

// x warmth, y brightness, z contrast, w saturation; each -1..1, 0 is no change.
const GRADE = `
uniform vec4 grade;
vec3 applyGrade(vec3 c) {
  c += grade.y * 0.25;
  c = (c - 0.5) * (1.0 + grade.z * 0.6) + 0.5;
  c = mix(vec3(dot(c, LUMA)), c, 1.0 + grade.w);
  c += vec3(grade.x, grade.x * 0.25, -grade.x) * 0.08;
  return clamp(c, 0.0, 1.0);
}`;

// Cartoon style, for the whole frame. toonA and toonB are the camera frame
// blurred twice, B wider than A. A is the flat colour: pores and fine texture
// gone. Where A is darker than B there is a dark edge, and that becomes an
// ink line (a difference of Gaussians). Shading is cut into four soft bands.
// ink scales the lines: the edges are the camera's, so a blurred or replaced
// background must not show the real background's outlines.
const TOON = `
uniform float toonAmt;
uniform sampler2D toonT;
vec3 toonBase(vec3 c, vec2 p) {
  return toonAmt > 0.0 ? mix(c, texture2D(toonT, p).rgb, toonAmt) : c;
}
vec3 applyToon(vec3 c, vec2 p, float ink) {
  if (toonAmt <= 0.0) return c;
  float l = max(dot(c, LUMA), 0.002);
  float f = l * 4.0 + 0.35;
  float q = (floor(f) + smoothstep(0.3, 0.7, fract(f))) / 4.0;
  vec3 t = c * min(q / l, 3.0);
  t = mix(vec3(dot(t, LUMA)), t, 1.3);
  vec4 ab = texture2D(toonT, p);
  t *= 1.0 - 0.82 * ink * smoothstep(0.012, 0.035, ab.a - dot(ab.rgb, LUMA));
  return mix(c, clamp(t, 0.0, 1.0), toonAmt);
}`;

// Recolour keeping the light and shade of the original. lift is the
// brightness of the darkest pixel: lips keep their shading at 0.35, but an
// iris is near black, and at that lift a blue iris would stay near black.
const TINT = `
vec3 tint(vec3 c, vec3 t, float amt, float lift) {
  vec3 col = t * (lift + (1.75 - lift) * dot(c, LUMA));
  return mix(c, clamp(col, 0.0, 1.0), amt);
}`;

// Hair colour, from the segmentation mask (hair in red, background in
// green), sampled at the pixel's place in the camera image. Both passes call
// it the same way, so hair the face mesh covers and hair outside it match.
const HAIR = `
uniform sampler2D segT;
uniform float hairAmt;
uniform vec3 hairColor;
vec3 applyHair(vec3 c, vec2 p) {
  return hairAmt > 0.0 ? tint(c, hairColor, hairAmt * texture2D(segT, p).r, ${HAIR_LIFT.toFixed(2)}) : c;
}`;

// The camera image, full frame, for the background.
const VS_BG = `
attribute vec2 p;
uniform float mirror;
varying vec2 uv;
void main() {
  uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (mirror > 0.5) uv.x = 1.0 - uv.x;
  gl_Position = vec4(p, 0.0, 1.0);
}`;
// The background around the face, as a grid: src is where a grid point is in
// the camera image, pos where it is drawn, both 0..1 image coordinates.
const VS_BGW = `
attribute vec2 pos;
attribute vec2 src;
uniform float mirror;
varying vec2 uv;
void main() {
  float x = mirror > 0.5 ? 1.0 - pos.x : pos.x;
  gl_Position = vec4(x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
  uv = src;
}`;
// Hair is recoloured first and the background goes behind the result, as a
// person in front of a backdrop. The cartoon style and the grade come after
// both, over the whole frame, so the mask's edge never shows as a seam.
const FS_BG = `${PRECISION}${GRADE}${TOON}${TINT}${HAIR}
uniform sampler2D tex;
uniform sampler2D bgLow;
uniform float bgBlur;
uniform float bgAmt;
uniform vec3 bgColor;
varying vec2 uv;
void main() {
  vec3 c = applyHair(toonBase(texture2D(tex, uv).rgb, uv), uv);
  float fx = 0.0;
  if (bgBlur > 0.0 || bgAmt > 0.0) {
    float bg = texture2D(segT, uv).g;
    // The blur's first quarter fades it in: even the smallest blur is a
    // quarter-size image, and switching to it at once would be a jump.
    float bw = min(bgBlur * 4.0, 1.0);
    if (bgBlur > 0.0) {
      // The background alone, blurred: colour times mask over the mask.
      vec4 b = texture2D(bgLow, uv);
      c = mix(c, b.rgb / max(b.a, 1.0 / 255.0), bg * bw);
    }
    c = mix(c, bgColor, bg * bgAmt);
    fx = bg * max(bgBlur > 0.0 ? bw : 0.0, bgAmt);
  }
  gl_FragColor = vec4(applyGrade(applyToon(c, uv, 1.0 - fx)), 1.0);
}`;

// Offscreen passes keep image orientation: texture row t holds image row y = t,
// the same as an uploaded image, so every texture is sampled with landmark
// coordinates directly.
const VS_QUAD = `
attribute vec2 p;
varying vec2 uv;
void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`;
const FS_COPY = `${PRECISION}
uniform sampler2D tex;
varying vec2 uv;
void main() { gl_FragColor = texture2D(tex, uv); }`;
// The cartoon style's two blurs in one texture: A's colour, B's brightness.
const FS_PACK = `${PRECISION}
uniform sampler2D tex;
uniform sampler2D texB;
varying vec2 uv;
void main() { gl_FragColor = vec4(texture2D(tex, uv).rgb, dot(texture2D(texB, uv).rgb, LUMA)); }`;
// The camera frame times the background mask, with the mask in alpha. Blurred
// and divided by its alpha, it is the background blurred on its own: a plain
// blur of the frame would smear the person's outline into it as a halo.
const FS_BGMASK = `${PRECISION}
uniform sampler2D tex;
uniform sampler2D segT;
varying vec2 uv;
void main() {
  float m = texture2D(segT, uv).g;
  gl_FragColor = vec4(texture2D(tex, uv).rgb * m, m);
}`;
// One direction of a 13-tap Gaussian; dir is the tap spacing in texture units.
const FS_BLUR = `${PRECISION}
uniform sampler2D tex;
uniform vec2 dir;
varying vec2 uv;
void main() {
  vec4 sum = vec4(0.0);
  float wsum = 0.0;
  for (int i = -6; i <= 6; i++) {
    float f = float(i);
    float w = exp(-f * f / 18.0);
    sum += texture2D(tex, uv + dir * f) * w;
    wsum += w;
  }
  gl_FragColor = sum / wsum;
}`;

// The face, through the mesh. pos is where a point is drawn (after the eye
// warp), lpos where it is in the camera image, tc where it is in the
// reference, and fpos where it is in the live face's own frame (faceFrame).
const VS_FACE = `
attribute vec2 pos;
attribute vec2 lpos;
attribute vec2 tc;
attribute vec2 fpos;
attribute float a;
uniform float mirror;
varying vec2 dp;
varying vec2 lp;
varying vec2 uv;
varying vec2 fc;
varying float va;
void main() {
  float x = mirror > 0.5 ? 1.0 - pos.x : pos.x;
  gl_Position = vec4(x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
  dp = pos;
  lp = lpos;
  uv = tc;
  fc = fpos;
  va = a;
}`;
const FS_FACE = `${PRECISION}${GRADE}${TOON}${TINT}${HAIR}
uniform sampler2D video;
uniform sampler2D liveLow;
uniform sampler2D liveSmooth;
uniform sampler2D refLow;
// The transfer mask in colour, the skin mask in alpha (packed in draw()).
uniform sampler2D masks;
uniform sampler2D maskX;
uniform vec2 vsize;
uniform float bgCut;
uniform float hasRef;
uniform float blend;
uniform float swapMode;
uniform float lumNorm;
uniform float irisOn;
uniform vec3 refIris;
uniform float smoothAmt;
uniform float lipAmt;
uniform float eyeAmt;
uniform vec3 lipColor;
uniform vec3 eyeColor;
uniform vec3 iris0;
uniform vec3 iris1;
// The filters' amounts: anime, feminine, masculine.
uniform vec3 filt;
varying vec2 dp;
varying vec2 lp;
varying vec2 uv;
varying vec2 fc;
varying float va;

// The iris as a ring: the pupil in the middle stays dark.
float irisRing(vec3 ir, vec2 px) {
  float d = distance(px, ir.xy);
  // smoothstep needs edge0 < edge1 (reversed edges are undefined in GLSL ES),
  // so the outer fade is written as 1 - smoothstep.
  return (1.0 - smoothstep(ir.z * 0.9, ir.z * 1.08, d)) * smoothstep(ir.z * 0.3, ir.z * 0.45, d);
}

// A soft spot on both sides of the face at once, in the face's own frame:
// the same place on the face at any size, position or roll.
float spot(vec2 c, vec2 r) {
  vec2 q = (vec2(abs(fc.x), fc.y) - c) / r;
  return exp(-dot(q, q));
}

// Value noise with no texture: a hash of the cell corners, smoothly mixed.
float hash(vec2 p) {
  vec3 p3 = fract(p.xyx * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

// A drawn catchlight: a small white disc up and to one side in the iris.
float catchlight(vec3 ir, vec2 px) {
  return 1.0 - smoothstep(ir.z * 0.15, ir.z * 0.25, distance(px, ir.xy + vec2(-0.32, -0.34) * ir.z));
}

void main() {
  vec3 c = toonBase(texture2D(video, lp).rgb, lp);

  if (smoothAmt > 0.0) {
    vec3 sm = texture2D(liveSmooth, lp).rgb;
    // Edge-aware: where the blurred image differs a lot, there is an edge
    // (a wrinkle line stays soft, a jawline or nostril stays sharp).
    float keep = 1.0 - smoothstep(0.06, 0.2, distance(c, sm));
    c = mix(c, sm, smoothAmt * texture2D(masks, dp).a * keep);
  }

  // Lips in red, eye openings in green, brows (masculine filter) in blue.
  vec3 feat = texture2D(maskX, dp).rgb;
  vec2 px = dp * vsize;
  float iris = max(irisRing(iris0, px), irisRing(iris1, px));

  if (hasRef > 0.5) {
    // Take the photo's colour, never its pixels. Both images are blurred to a
    // scale where makeup, lipstick, blush and skin tone survive but eyes,
    // pores and the shape of a nose do not, and the live face is recoloured
    // by their ratio. The live face keeps its own features, detail and
    // expression; in a face swap the mesh has already been moved to the
    // photo's proportions, so the photo's colours land on matching features.
    vec3 rl = texture2D(refLow, uv).rgb;
    vec3 ll = texture2D(liveLow, lp).rgb;
    float lr = max(dot(rl, LUMA), 0.03);
    float lv = max(dot(ll, LUMA), 0.03);
    vec3 chroma = (max(rl, vec3(0.01)) / lr) / (max(ll, vec3(0.01)) / lv);
    // Look: darkness (eyeshadow, contour) transfers at half strength,
    // measured against the faces' average brightness, so a darker photo does
    // not darken the live face. Face swap: the photo's light and dark come
    // over in full, less half the difference in overall exposure, so the
    // face is the photo's but does not glow or sink against the background.
    float shade = swapMode > 0.5 ? clamp(lr / lv / sqrt(lumNorm), 0.4, 2.5)
      : sqrt(clamp(lr / lv / lumNorm, 0.6, 1.5));
    vec3 cr = swapMode > 0.5 ? clamp(chroma, 0.3, 3.0) : clamp(chroma, 0.4, 2.5);
    vec3 looked = clamp(c * cr * shade, 0.0, 1.0);
    // The eye openings are left out: a gaze that differs from the photo's
    // would put its iris colour on the whites. The iris gets the photo's
    // iris colour on the live iris instead.
    c = mix(c, looked, blend * texture2D(masks, dp).r * (1.0 - feat.g));
    if (irisOn > 0.5) c = tint(c, refIris, blend * feat.g * iris, 0.7);
  }

  if (lipAmt > 0.0) c = tint(c, lipColor, lipAmt * feat.r, 0.35);
  if (eyeAmt > 0.0) c = tint(c, eyeColor, eyeAmt * feat.g * iris, 0.7);

  // The filters' colour. A lip colour chosen in Edits wins over theirs.
  if (filt.x + filt.y > 0.0) {
    // Blush on the cheeks, eyes and lips left out.
    float cheek = spot(vec2(0.72, 0.52), vec2(0.25, 0.17)) * (1.0 - feat.r) * (1.0 - feat.g);
    c = mix(c, c * vec3(1.0, 0.72, 0.76), cheek * (0.5 * filt.x + 0.32 * filt.y));
  }
  if (filt.x > 0.0) {
    // Anime: pink lips, and a catchlight in each eye.
    if (lipAmt <= 0.0) c = tint(c, vec3(0.86, 0.42, 0.48), 0.3 * filt.x * feat.r, 0.35);
    c = mix(c, vec3(1.0), 0.9 * filt.x * feat.g * max(catchlight(iris0, px), catchlight(iris1, px)));
  }
  if (filt.y > 0.0) {
    // Feminine: rose lips, a soft eyeshadow on the upper lids, darker lashes
    // with a small wing (blue in the mask), and a soft glow on the skin.
    if (lipAmt <= 0.0) c = tint(c, vec3(0.8, 0.32, 0.4), 0.38 * filt.y * feat.r, 0.35);
    c = mix(c, c * vec3(0.84, 0.72, 0.78), 0.45 * filt.y * spot(vec2(0.55, -0.15), vec2(0.24, 0.08)) * (1.0 - feat.g));
    c = mix(c, c * 0.3, 0.75 * filt.y * feat.b * (1.0 - feat.g));
    c = mix(c, 1.0 - (1.0 - c) * (1.0 - c), 0.16 * filt.y * texture2D(masks, dp).a);
  }
  if (filt.z > 0.0) {
    // Masculine: darker, heavier brows, a shadow under the brow ridge, and
    // stubble over the jaw, chin and upper lip: a shadow with a grain of
    // dots fixed to the face, so it moves with the skin.
    c = mix(c, c * 0.5, 0.55 * filt.z * feat.b);
    c *= 1.0 - 0.2 * filt.z * spot(vec2(0.55, -0.24), vec2(0.28, 0.07)) * (1.0 - feat.g);
    float ax = abs(fc.x);
    float edge = ax < 0.45 ? mix(0.76, 0.86, ax / 0.45) : ax < 0.75 ? mix(0.86, 0.92, (ax - 0.45) / 0.3) : mix(0.92, 0.55, min((ax - 0.75) / 0.45, 1.0));
    float beard = smoothstep(edge - 0.03, edge + 0.08, fc.y) * (1.0 - feat.r);
    float grain = smoothstep(0.4, 0.8, noise(fc * 95.0));
    c = mix(c, c * vec3(0.5, 0.5, 0.56), beard * filt.z * (0.26 + 0.38 * grain));
  }
  c = applyHair(c, lp);

  // Where the mesh reaches past the face onto a blurred or replaced
  // background, the face fades out and the background pass shows instead.
  float a = va * (1.0 - bgCut * texture2D(segT, lp).g);
  gl_FragColor = vec4(applyGrade(applyToon(c, lp, 1.0)), a);
}`;

const VS_LINE = `
attribute vec2 pos;
uniform float mirror;
void main() {
  float x = mirror > 0.5 ? 1.0 - pos.x : pos.x;
  gl_Position = vec4(x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
}`;
const FS_LINE = `
precision mediump float;
uniform vec4 color;
void main() { gl_FragColor = color; }`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  for (const [type, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    gl.attachShader(p, s);
  }
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  p.loc = new Map();
  return p;
}

function texture(gl) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}

export class Renderer {
  constructor(canvas, mesh) {
    // preserveDrawingBuffer keeps the last frame readable for capture. alpha:
    // false because blending also writes destination alpha; on an alpha
    // canvas the blended face would show the page through it.
    const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true, alpha: false });
    if (!gl) throw new Error('WebGL is not available');
    if (!shapes) shapes = buildShapes();
    this.gl = gl;
    this.mesh = mesh;
    this.bg = program(gl, VS_BG, FS_BG);
    this.bgWarp = program(gl, VS_BGW, FS_BG);
    this.copyProg = program(gl, VS_QUAD, FS_COPY);
    this.blurProg = program(gl, VS_QUAD, FS_BLUR);
    this.packProg = program(gl, VS_QUAD, FS_PACK);
    this.bgMaskProg = program(gl, VS_QUAD, FS_BGMASK);
    this.face = program(gl, VS_FACE, FS_FACE);
    this.line = program(gl, VS_LINE, FS_LINE);
    this.videoTex = texture(gl);
    this.refTex = texture(gl);
    this.featureTex = texture(gl);
    this.targets = {};

    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    this.posBuf = gl.createBuffer();
    this.lposBuf = gl.createBuffer();
    this.tcBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tcBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(MESH_POINTS * 2), gl.STATIC_DRAW);
    this.alphaBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.alphaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.alpha, gl.STATIC_DRAW);
    this.triBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.tris, gl.STATIC_DRAW);
    this.lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.lines, gl.STATIC_DRAW);
    this.fposBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.fposBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(MESH_POINTS * 2), gl.DYNAMIC_DRAW);

    // The background grid: (GRID + 1)^2 points, two triangles a cell.
    const idx = [];
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        const a = r * (GRID + 1) + c;
        idx.push(a, a + 1, a + GRID + 1, a + 1, a + GRID + 2, a + GRID + 1);
      }
    }
    this.gridIdx = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.gridIdx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW);
    this.gridCount = idx.length;
    this.gridSrc = new Float32Array((GRID + 1) ** 2 * 2);
    this.gridPos = new Float32Array((GRID + 1) ** 2 * 2);
    this.gridSrcBuf = gl.createBuffer();
    this.gridPosBuf = gl.createBuffer();
    this.ovalPx = new Float32Array(mesh.oval.length * 4);

    this.disp = new Float32Array(ALL_POINTS * 2);
    this.lpos = new Float32Array(MESH_POINTS * 2);
    this.liveLocal = new Float32Array(ALL_POINTS * 2);
    this.refLocal = null;
    this.rest = null;
    this.shapeKey = '';
    this.canvases = { transfer: document.createElement('canvas'), skin: document.createElement('canvas'),
      feature: document.createElement('canvas') };
    this.hasRef = false;
    this.refHair = null;
    this.segReady = false;
    this.enabled = new Set();
  }

  u(prog, name) {
    if (!prog.loc.has(name)) prog.loc.set(name, this.gl.getUniformLocation(prog, name));
    return prog.loc.get(name);
  }

  attrib(prog, name, buf, size) {
    const gl = this.gl;
    const loc = gl.getAttribLocation(prog, name);
    if (loc < 0) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    this.enabled.add(loc);
  }

  // An attribute left enabled from the previous program still bounds-checks
  // against its old buffer: the 4-vertex quad would fail a draw indexed to 467.
  useProgram(prog) {
    for (const loc of this.enabled) this.gl.disableVertexAttribArray(loc);
    this.enabled.clear();
    this.gl.useProgram(prog);
  }

  /** A render target of this size, made or remade on demand. */
  target(name, w, h) {
    const gl = this.gl;
    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));
    let t = this.targets[name];
    if (t && t.w === w && t.h === h) return t;
    if (!t) t = this.targets[name] = { tex: texture(gl), fb: gl.createFramebuffer() };
    t.w = w;
    t.h = h;
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, t.fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return t;
  }

  quadPass(prog, src, dst) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
    gl.viewport(0, 0, dst.w, dst.h);
    this.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, src);
    gl.uniform1i(this.u(prog, 'tex'), 0);
    this.attrib(prog, 'p', this.quad, 2);
  }

  copy(src, dst) {
    this.quadPass(this.copyProg, src, dst);
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  /** Gaussian blur of `t` in place, sigma in t's own pixels. `tmp` must match its size. */
  blur(t, tmp, sigma) {
    if (sigma < 0.3) return;
    const step = sigma / 3;
    const gl = this.gl;
    this.quadPass(this.blurProg, t.tex, tmp);
    gl.uniform2f(this.u(this.blurProg, 'dir'), step / t.w, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    this.quadPass(this.blurProg, tmp.tex, t);
    gl.uniform2f(this.u(this.blurProg, 'dir'), 0, step / t.h);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /** Upload a mask canvas into a target texture of the same size. */
  upload(canvas, t) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
  }

  /**
   * The reference photo and its 478 points. Only a blurred copy is kept for
   * drawing: the photo's colours, never its pixels. Also its points in the
   * face's own frame, for the face swap's shape, and the mean iris colour.
   */
  setReference(image, points) {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.bindTexture(gl.TEXTURE_2D, this.refTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    const tc = new Float32Array(MESH_POINTS * 2);
    for (let i = 0; i < MESH_POINTS; i++) { tc[i * 2] = points[i].x; tc[i * 2 + 1] = points[i].y; }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tcBuf);
    gl.bufferData(gl.ARRAY_BUFFER, tc, gl.STATIC_DRAW);

    const W = image.width, H = image.height;
    const fw = faceWidth(points, W, H);
    const look = this.target('refLook', W / 4, H / 4);
    this.copy(this.refTex, look);
    this.blur(look, this.target('refTmp', W / 4, H / 4), fw * LOOK_BLUR / 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.refIris = irisColor(image, points);
    this.refLocal = points.length >= ALL_POINTS
      ? toLocal(flatten(points, new Float32Array(ALL_POINTS * 2)), W, H, new Float32Array(ALL_POINTS * 2)) : null;
    this.hasRef = true;
  }

  clearReference() { this.hasRef = false; this.refLocal = null; this.refHair = null; }

  /** The photo's hair tint from referenceHair(), or null. Used in a face swap. */
  setReferenceHair(t) { this.refHair = t; }

  /**
   * The hair and background masks of the latest segmented frame, from
   * segmentFrame(). Feathered here, and kept until the next one, since the
   * segmenter may skip frames.
   */
  setSegmentation({ data, w, h }) {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    const t = this.target('seg', w, h);
    gl.bindTexture(gl.TEXTURE_2D, t.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    this.blur(t, this.target('segTmp', w, h), SEG_FEATHER);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.segReady = true;
  }

  clearSegmentation() { this.segReady = false; }

  /** Per-point shape weight for the chosen areas: the most any of them allows. */
  shapeWeights(regions) {
    const key = regions.join();
    if (key !== this.shapeKey) {
      this.shapeKey = key;
      this.weights = new Float32Array(ALL_POINTS);
      for (const r of regions) {
        const w = this.mesh.shape[r];
        for (let i = 0; i < ALL_POINTS; i++) this.weights[i] = Math.max(this.weights[i], w[i]);
      }
    }
    return this.weights;
  }

  /**
   * Face swap shape: move the live points toward the photo's proportions.
   * Each point moves by the difference between where the photo has it and
   * where this face has it at rest, both in the face's own frame. The rest
   * shape is a slow average of the live face, so a blink or a smile is not
   * part of it: the live expression still moves the face, on the photo's
   * proportions (an anime face keeps its big eyes and still blinks).
   */
  morph(d, regions, amount, W, H) {
    const L = toLocal(d, W, H, this.liveLocal);
    if (!this.rest) this.rest = L.slice();
    else for (let i = 0; i < L.length; i++) this.rest[i] += (L[i] - this.rest[i]) * 0.04;
    const w = this.shapeWeights(regions);
    const { ox, oy, ux, uy, s } = faceFrame(d, W, H);
    const R = this.refLocal, rest = this.rest;
    for (let i = 0; i < ALL_POINTS; i++) {
      const k = amount * w[i];
      if (k <= 0) continue;
      const lx = L[i * 2] + k * (R[i * 2] - rest[i * 2]);
      const ly = L[i * 2 + 1] + k * (R[i * 2 + 1] - rest[i * 2 + 1]);
      d[i * 2] = (ox + s * (lx * ux - ly * uy)) / W;
      d[i * 2 + 1] = (oy + s * (lx * uy + ly * ux)) / H;
    }
  }

  /**
   * Keep the outline's movement under EDGE_MAX face widths, over the outline
   * and the three rings in from it, so the background can follow without
   * folding. Smooth: half the limit is kept to 2%, and it never goes over.
   */
  capEdge(d, fw, W, H) {
    const c = EDGE_MAX * fw;
    for (const i of this.mesh.edge) {
      const dx = (d[i * 2] - this.lpos[i * 2]) * W, dy = (d[i * 2 + 1] - this.lpos[i * 2 + 1]) * H;
      const f = 1 / Math.sqrt(Math.sqrt(1 + (Math.hypot(dx, dy) / c) ** 4));
      d[i * 2] = this.lpos[i * 2] + dx * f / W;
      d[i * 2 + 1] = this.lpos[i * 2 + 1] + dy * f / H;
    }
  }

  /**
   * The background grid around the face, moved with the outline. A grid
   * point moves by the Gaussian-weighted mean of the outline points'
   * movement, divided by no less than the weight an outline point gets from
   * its neighbours: on the outline it moves as the outline does, and four
   * sigma away it does not move. Fills gridSrc and gridPos; false when the
   * outline did not move.
   */
  warpBackground(d, fw, W, H) {
    const o = this.mesh.oval, n = o.length, q = this.ovalPx;
    const cap = EDGE_MAX * fw;
    let moved = 0, x0 = W, y0 = H, x1 = 0, y1 = 0;
    for (let k = 0; k < n; k++) {
      const i = o[k];
      const x = this.lpos[i * 2] * W, y = this.lpos[i * 2 + 1] * H;
      let dx = d[i * 2] * W - x, dy = d[i * 2 + 1] * H - y;
      const m = Math.hypot(dx, dy);
      if (m > cap) { dx *= cap / m; dy *= cap / m; }
      q[k * 4] = x; q[k * 4 + 1] = y; q[k * 4 + 2] = dx; q[k * 4 + 3] = dy;
      moved = Math.max(moved, m);
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    if (moved < 0.5) return false;
    const sig = BG_SIGMA * fw, e = -1 / (2 * sig * sig);
    let w0 = Infinity;
    for (let a = 0; a < n; a++) {
      let sum = 0;
      for (let b = 0; b < n; b++) sum += Math.exp(((q[a * 4] - q[b * 4]) ** 2 + (q[a * 4 + 1] - q[b * 4 + 1]) ** 2) * e);
      w0 = Math.min(w0, sum);
    }
    const pad = 4 * sig;
    x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
    x1 = Math.min(W, x1 + pad); y1 = Math.min(H, y1 + pad);
    for (let r = 0, j = 0; r <= GRID; r++) {
      const gy = y0 + (y1 - y0) * r / GRID;
      for (let c = 0; c <= GRID; c++, j += 2) {
        const gx = x0 + (x1 - x0) * c / GRID;
        let sw = 0, sx = 0, sy = 0;
        for (let k = 0; k < n; k++) {
          const w = Math.exp(((gx - q[k * 4]) ** 2 + (gy - q[k * 4 + 1]) ** 2) * e);
          sw += w; sx += w * q[k * 4 + 2]; sy += w * q[k * 4 + 3];
        }
        const div = Math.max(sw, w0);
        this.gridSrc[j] = gx / W;
        this.gridSrc[j + 1] = gy / H;
        this.gridPos[j] = (gx + sx / div) / W;
        this.gridPos[j + 1] = (gy + sy / div) / H;
      }
    }
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridSrcBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.gridSrc, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridPosBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.gridPos, gl.DYNAMIC_DRAW);
    return true;
  }

  /**
   * One frame.
   * f: { mirror, points (478 live points or null), lumNorm, s (settings) }
   * s: { regions[], blend, refMode ('look'|'features'), refHair, smooth, eyeSize,
   *      eyeColor [r,g,b], eyeAmt, lipColor [r,g,b], lipAmt, toon,
   *      hairColor [r,g,b], hairAmt, bgBlur, bgColor [r,g,b], bgAmt,
   *      warmth, brightness, contrast, saturation, filter, filterAmt, showMesh }
   */
  draw(video, f) {
    const gl = this.gl;
    const s = f.s;
    const W = video.videoWidth, H = video.videoHeight;
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    const pts = f.points?.length >= ALL_POINTS ? f.points : null;
    const transfer = this.hasRef && s.blend > 0 && s.regions.length > 0;
    const swap = transfer && s.refMode === 'features' && this.refLocal;
    // A filter adds to the edits, it does not replace them: anime and
    // feminine smooth the skin at least this much, anime draws in the cartoon
    // style at least this much, and masculine smooths less than asked.
    const fa = SHAPES[s.filter] ? s.filterAmt : 0;
    const anime = s.filter === 'anime' ? fa : 0;
    const fem = s.filter === 'feminine' ? fa : 0;
    const masc = s.filter === 'masculine' ? fa : 0;
    const smooth = Math.max(s.smooth, 0.7 * anime, 0.5 * fem) * (1 - 0.7 * masc);
    const toonAmt = Math.max(s.toon, 0.85 * anime);
    // Hair and background need a mask; until the segmenter has one, neither
    // shows. In a face swap the photo's hair colour, if asked for, replaces
    // the chosen one, at the swap's strength.
    const seg = this.segReady;
    const refHair = swap && s.refHair && this.refHair;
    const hairAmt = !seg ? 0 : refHair ? s.blend : s.hairAmt;
    const hairColor = refHair || s.hairColor;
    const bgBlur = seg ? s.bgBlur : 0, bgAmt = seg ? s.bgAmt : 0;
    let irisA = [0, 0, 0], irisB = [0, 0, 0];
    // With no face in view the cartoon style still needs a scale to work at.
    let fw = W / 4;
    let bgWarped = false;
    if (!pts) this.rest = null;
    if (pts) {
      const d = flatten(pts, this.disp);
      for (let i = 0; i < MESH_POINTS * 2; i++) this.lpos[i] = d[i];
      fw = faceWidth(pts, W, H);
      // The live points in the face's own frame: the filter's field, and the
      // face pass's place for the filter's colour.
      const L = toLocal(d, W, H, this.liveLocal);
      const fr = faceFrame(d, W, H);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.fposBuf);
      gl.bufferData(gl.ARRAY_BUFFER, L.subarray(0, MESH_POINTS * 2), gl.DYNAMIC_DRAW);
      if (swap) this.morph(d, s.regions, s.blend, W, H);
      // morph() leaves liveLocal as it found it: the live points.
      if (fa > 0) reshape(d, L, fr, this.mesh.shape.brows, SHAPES[s.filter], fa, W, H);
      // Only a filter moves the outline (a face swap keeps it, see buildMesh).
      if (fa > 0) this.capEdge(d, fw, W, H);
      displace(d, s.eyeSize, W, H);
      if (fa > 0) bgWarped = this.warpBackground(d, fw, W, H);
      const Pq = i => [d[i * 2] * W / 4, d[i * 2 + 1] * H / 4];
      const Ph = i => [d[i * 2] * W / 2, d[i * 2 + 1] * H / 2];

      if (transfer) {
        const low = this.target('liveLook', W / 4, H / 4);
        this.copy(this.videoTex, low);
        this.blur(low, this.target('tmpQ', W / 4, H / 4), fw * LOOK_BLUR / 4);
      }
      if (smooth > 0) {
        const sm = this.target('liveSmooth', W / 2, H / 2);
        this.copy(this.videoTex, sm);
        this.blur(sm, this.target('tmpH', W / 2, H / 2), fw / 70 / 2);
        const m = this.target('maskS', W / 4, H / 4);
        drawSkinMask(this.canvases.skin, m.w, m.h, Pq, fw / 4);
        this.upload(this.canvases.skin, m);
        this.blur(m, this.target('tmpQ', W / 4, H / 4), 0.02 * fw / 4);
      }
      if (transfer) {
        const m = this.target('maskT', W / 4, H / 4);
        drawTransferMask(this.canvases.transfer, m.w, m.h, Pq, fw / 4, s.regions);
        this.upload(this.canvases.transfer, m);
        this.blur(m, this.target('tmpQ', W / 4, H / 4), 0.022 * fw / 4);
      }
      if (transfer || smooth > 0) {
        // Both masks in one texture, transfer in colour and skin in alpha, so
        // the face pass has a texture unit free for the segmentation mask.
        this.quadPass(this.packProg, transfer ? this.targets.maskT.tex : null, this.target('masks', W / 4, H / 4));
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, smooth > 0 ? this.targets.maskS.tex : null);
        gl.uniform1i(this.u(this.packProg, 'texB'), 1);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.activeTexture(gl.TEXTURE0);
      }
      if (s.lipAmt > 0 || s.eyeAmt > 0 || transfer || fa > 0) {
        const c = this.canvases.feature;
        drawFeatureMask(c, Math.round(W / 2), Math.round(H / 2), Ph, masc > 0 ? fw / 2 : 0, fem > 0 ? fw / 2 : 0);
        gl.bindTexture(gl.TEXTURE_2D, this.featureTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
      }
      irisA = iris(d, 468, W, H);
      irisB = iris(d, 473, W, H);
    }

    // Cartoon style: the flat-colour and the edge blurs, at half size, packed
    // into one texture (B's brightness in A's alpha). The face pass already
    // uses seven texture units, and WebGL promises only eight.
    const T = this.targets;
    if (toonAmt > 0) {
      const tmp = this.target('tmpH', W / 2, H / 2);
      const a = this.target('toonA', W / 2, H / 2);
      const b = this.target('toonB', W / 2, H / 2);
      this.copy(this.videoTex, a);
      this.blur(a, tmp, fw / 90 / 2);
      this.copy(a.tex, b);
      this.blur(b, tmp, fw / 90 / 2 * 1.25);
      this.quadPass(this.packProg, a.tex, this.target('toon', W / 2, H / 2));
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, b.tex);
      gl.uniform1i(this.u(this.packProg, 'texB'), 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.activeTexture(gl.TEXTURE0);
    }
    const toon = (prog, unit) => {
      gl.uniform1f(this.u(prog, 'toonAmt'), toonAmt);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, toonAmt > 0 ? T.toon.tex : null);
      gl.uniform1i(this.u(prog, 'toonT'), unit);
      gl.activeTexture(gl.TEXTURE0);
    };
    const hair = (prog, unit) => {
      gl.uniform1f(this.u(prog, 'hairAmt'), hairAmt);
      gl.uniform3fv(this.u(prog, 'hairColor'), hairColor);
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, seg ? T.seg.tex : null);
      gl.uniform1i(this.u(prog, 'segT'), unit);
      gl.activeTexture(gl.TEXTURE0);
    };

    // Background blur, at quarter size, in two rounds: one Gaussian this
    // wide would space its 13 taps too far apart and show them. Its width
    // is a share of the frame, not of the face: it is the backdrop's blur.
    if (bgBlur > 0) {
      const low = this.target('bgLow', W / 4, H / 4);
      const tmp = this.target('tmpQ', W / 4, H / 4);
      this.quadPass(this.bgMaskProg, this.videoTex, low);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, T.seg.tex);
      gl.uniform1i(this.u(this.bgMaskProg, 'segT'), 1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.activeTexture(gl.TEXTURE0);
      const sigma = bgBlur * Math.max(W, H) / 32 / 4 * Math.SQRT1_2;
      this.blur(low, tmp, sigma);
      this.blur(low, tmp, sigma);
    }

    // Background: the whole camera frame, with its hair and background
    // effects, graded. Where the outline moved, the grid around the face is
    // drawn over it, moved with the outline.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    const grade = [s.warmth, s.brightness, s.contrast, s.saturation];
    const background = prog => {
      this.useProgram(prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
      gl.uniform1i(this.u(prog, 'tex'), 0);
      gl.uniform1f(this.u(prog, 'mirror'), f.mirror ? 1 : 0);
      gl.uniform4fv(this.u(prog, 'grade'), grade);
      toon(prog, 1);
      hair(prog, 2);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, bgBlur > 0 ? T.bgLow.tex : null);
      gl.uniform1i(this.u(prog, 'bgLow'), 3);
      gl.activeTexture(gl.TEXTURE0);
      gl.uniform1f(this.u(prog, 'bgBlur'), bgBlur);
      gl.uniform1f(this.u(prog, 'bgAmt'), bgAmt);
      gl.uniform3fv(this.u(prog, 'bgColor'), s.bgColor);
    };
    background(this.bg);
    this.attrib(this.bg, 'p', this.quad, 2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (bgWarped) {
      background(this.bgWarp);
      this.attrib(this.bgWarp, 'pos', this.gridPosBuf, 2);
      this.attrib(this.bgWarp, 'src', this.gridSrcBuf, 2);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.gridIdx);
      gl.drawElements(gl.TRIANGLES, this.gridCount, gl.UNSIGNED_SHORT, 0);
    }
    if (!pts) return;

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.disp.subarray(0, MESH_POINTS * 2), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.lposBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.lpos, gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    // The face: every edit in one pass.
    const p = this.face;
    this.useProgram(p);
    const units = [
      ['video', this.videoTex], ['liveLow', T.liveLook?.tex], ['liveSmooth', T.liveSmooth?.tex],
      ['refLow', T.refLook?.tex], ['masks', T.masks?.tex], ['maskX', this.featureTex],
    ];
    units.forEach(([name, tex], i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, tex || null);
      gl.uniform1i(this.u(p, name), i);
    });
    // Eight units in all, the most WebGL promises.
    hair(p, units.length);
    toon(p, units.length + 1);
    gl.uniform1f(this.u(p, 'bgCut'), bgBlur > 0 || bgAmt > 0 ? 1 : 0);
    gl.uniform1f(this.u(p, 'mirror'), f.mirror ? 1 : 0);
    gl.uniform2f(this.u(p, 'vsize'), W, H);
    gl.uniform1f(this.u(p, 'hasRef'), transfer ? 1 : 0);
    gl.uniform1f(this.u(p, 'blend'), s.blend);
    gl.uniform1f(this.u(p, 'swapMode'), swap ? 1 : 0);
    gl.uniform1f(this.u(p, 'lumNorm'), f.lumNorm || 1);
    gl.uniform1f(this.u(p, 'irisOn'), transfer && (s.regions.includes('eyes') || s.regions.includes('face')) ? 1 : 0);
    gl.uniform3fv(this.u(p, 'refIris'), this.refIris || [0.3, 0.2, 0.15]);
    gl.uniform1f(this.u(p, 'smoothAmt'), smooth);
    gl.uniform3fv(this.u(p, 'filt'), [anime, fem, masc]);
    gl.uniform1f(this.u(p, 'lipAmt'), s.lipAmt);
    gl.uniform1f(this.u(p, 'eyeAmt'), s.eyeAmt);
    gl.uniform3fv(this.u(p, 'lipColor'), s.lipColor);
    gl.uniform3fv(this.u(p, 'eyeColor'), s.eyeColor);
    gl.uniform3fv(this.u(p, 'iris0'), irisA);
    gl.uniform3fv(this.u(p, 'iris1'), irisB);
    gl.uniform4fv(this.u(p, 'grade'), grade);
    this.attrib(p, 'pos', this.posBuf, 2);
    this.attrib(p, 'lpos', this.lposBuf, 2);
    this.attrib(p, 'tc', this.tcBuf, 2);
    this.attrib(p, 'fpos', this.fposBuf, 2);
    this.attrib(p, 'a', this.alphaBuf, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triBuf);
    gl.drawElements(gl.TRIANGLES, this.mesh.tris.length, gl.UNSIGNED_SHORT, 0);

    if (s.showMesh) {
      const l = this.line;
      this.useProgram(l);
      gl.uniform1f(this.u(l, 'mirror'), f.mirror ? 1 : 0);
      gl.uniform4fv(this.u(l, 'color'), [0.45, 0.9, 0.84, 0.55]);
      this.attrib(l, 'pos', this.posBuf, 2);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineBuf);
      gl.drawElements(gl.LINES, this.mesh.lines.length, gl.UNSIGNED_SHORT, 0);
    }
  }
}
