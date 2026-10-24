/* Face Cam — face tracking, the reference-photo transfer, and the edits.
 *
 * MediaPipe Face Landmarker gives 478 points per face: 468 mesh points plus
 * 10 iris points. The mesh points have the same meaning in every image —
 * point 1 is always the nose tip — so a reference photo and a live frame can
 * be matched point for point with no alignment step.
 *
 * Every frame, the live face is redrawn through the mesh. That one pass does
 * all the face work:
 *   - transfer: the chosen features (eyes, brows, nose, lips, or the whole
 *     face) come from the reference photo, cut by a soft mask so only those
 *     features change. In "live lighting" mode the reference keeps only its
 *     detail and takes its light and skin tone from the live face, so it sits
 *     in the scene instead of looking pasted on.
 *   - edits with no reference: skin smoothing, lip colour, eye colour, and
 *     eye size (a warp of the mesh points around each eye).
 *   - a colour grade (warmth, brightness, contrast, saturation), applied to
 *     the face and the background alike so there is no seam.
 *
 * Everything runs on the device. The model and WASM come from the CDN once and
 * the service worker keeps them; no frame or photo leaves the browser.
 */

export const TASKS_VERSION = '0.10.14';
const TASKS_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

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

let vision = null;
let FaceLandmarker = null;
let shapes = null;

async function loadTasks() {
  if (vision) return;
  const mod = await import(`${TASKS_BASE}/vision_bundle.mjs`);
  FaceLandmarker = mod.FaceLandmarker;
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

  return { tris: new Uint16Array(tris), lines: new Uint16Array(lines), alpha };
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

/** Hard-edged: red on the lips, green in the eye openings. Recolouring uses it. */
function drawFeatureMask(canvas, w, h, P) {
  const ctx = prepare(canvas, w, h);
  paint(ctx, 'lips', P, '#f00', 0, 'evenodd');
  paint(ctx, 'eyes', P, '#0f0', 0);
}

/**
 * Eye size: push points away from each iris centre, most near the centre and
 * not at all at 1.5 eye widths, so the rest of the face is untouched. The
 * push grows smoothly from zero, so the mesh never folds over itself.
 */
function displace(points, amount, W, H, out) {
  for (let i = 0; i < points.length; i++) { out[i * 2] = points[i].x; out[i * 2 + 1] = points[i].y; }
  if (amount <= 0 || points.length < ALL_POINTS) return out;
  const k = amount * 0.3;
  for (const [c, a, b] of [[468, 33, 133], [473, 362, 263]]) {
    const cx = points[c].x * W, cy = points[c].y * H;
    const R = 1.5 * Math.hypot((points[a].x - points[b].x) * W, (points[a].y - points[b].y) * H);
    for (let i = 0; i < points.length; i++) {
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

/** Iris centre and radius in pixels, from the five iris points starting at `c`. */
function iris(pts, c, W, H) {
  if (pts.length < ALL_POINTS) return [0, 0, 0];
  const cx = pts[c * 2] * W, cy = pts[c * 2 + 1] * H;
  let r = 0;
  for (let i = 1; i <= 4; i++) r += Math.hypot(pts[(c + i) * 2] * W - cx, pts[(c + i) * 2 + 1] * H - cy);
  return [cx, cy, r / 4];
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
const FS_BG = `${PRECISION}${GRADE}
uniform sampler2D tex;
varying vec2 uv;
void main() { gl_FragColor = vec4(applyGrade(texture2D(tex, uv).rgb), 1.0); }`;

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
// warp), lpos where it is in the camera image, tc where it is in the reference.
const VS_FACE = `
attribute vec2 pos;
attribute vec2 lpos;
attribute vec2 tc;
attribute float a;
uniform float mirror;
varying vec2 dp;
varying vec2 lp;
varying vec2 uv;
varying float va;
void main() {
  float x = mirror > 0.5 ? 1.0 - pos.x : pos.x;
  gl_Position = vec4(x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
  dp = pos;
  lp = lpos;
  uv = tc;
  va = a;
}`;
const FS_FACE = `${PRECISION}${GRADE}
uniform sampler2D video;
uniform sampler2D liveLow;
uniform sampler2D liveSmooth;
uniform sampler2D ref;
uniform sampler2D refLow;
uniform sampler2D maskT;
uniform sampler2D maskS;
uniform sampler2D maskX;
uniform vec2 vsize;
uniform float hasRef;
uniform float blend;
uniform float lighting;
uniform float smoothAmt;
uniform float lipAmt;
uniform float eyeAmt;
uniform vec3 gain;
uniform vec3 lipColor;
uniform vec3 eyeColor;
uniform vec3 iris0;
uniform vec3 iris1;
varying vec2 dp;
varying vec2 lp;
varying vec2 uv;
varying float va;

// Recolour keeping the light and shade of the original. lift is the
// brightness of the darkest pixel: lips keep their shading at 0.35, but an
// iris is near black, and at that lift a blue iris would stay near black.
vec3 tint(vec3 c, vec3 t, float amt, float lift) {
  vec3 col = t * (lift + (1.75 - lift) * dot(c, LUMA));
  return mix(c, clamp(col, 0.0, 1.0), amt);
}
// The iris as a ring: the pupil in the middle stays dark.
float irisRing(vec3 ir, vec2 px) {
  float d = distance(px, ir.xy);
  // smoothstep needs edge0 < edge1 (reversed edges are undefined in GLSL ES),
  // so the outer fade is written as 1 - smoothstep.
  return (1.0 - smoothstep(ir.z * 0.9, ir.z * 1.08, d)) * smoothstep(ir.z * 0.3, ir.z * 0.45, d);
}

void main() {
  vec3 c = texture2D(video, lp).rgb;

  if (smoothAmt > 0.0) {
    vec3 sm = texture2D(liveSmooth, lp).rgb;
    // Edge-aware: where the blurred image differs a lot, there is an edge
    // (a wrinkle line stays soft, a jawline or nostril stays sharp).
    float keep = 1.0 - smoothstep(0.06, 0.2, distance(c, sm));
    c = mix(c, sm, smoothAmt * texture2D(maskS, dp).r * keep);
  }

  if (hasRef > 0.5) {
    vec3 r = texture2D(ref, uv).rgb;
    vec3 rc;
    if (lighting > 0.5) {
      // Reference detail over live light: divide out the reference's own
      // low-frequency brightness and multiply in the live face's. Brightness
      // only: a per-channel ratio picks up strong local colour (eyeshadow,
      // lipstick) from the blur and tints the feature with its opposite.
      // Skin tone comes from gain instead, one face-wide correction.
      float rl = dot(texture2D(refLow, uv).rgb, LUMA);
      float ll = dot(texture2D(liveLow, lp).rgb, LUMA);
      rc = r * clamp(ll / max(rl, 0.03), 0.3, 3.0) * gain;
    } else {
      rc = r * gain;
    }
    c = mix(c, clamp(rc, 0.0, 1.0), blend * texture2D(maskT, dp).r);
  }

  vec2 feat = texture2D(maskX, dp).rg;
  if (lipAmt > 0.0) c = tint(c, lipColor, lipAmt * feat.r, 0.35);
  if (eyeAmt > 0.0) {
    vec2 px = dp * vsize;
    c = tint(c, eyeColor, eyeAmt * feat.g * max(irisRing(iris0, px), irisRing(iris1, px)), 0.7);
  }

  gl_FragColor = vec4(applyGrade(c), va);
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
    this.copyProg = program(gl, VS_QUAD, FS_COPY);
    this.blurProg = program(gl, VS_QUAD, FS_BLUR);
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

    this.disp = new Float32Array(ALL_POINTS * 2);
    this.lpos = new Float32Array(MESH_POINTS * 2);
    this.canvases = { transfer: document.createElement('canvas'), skin: document.createElement('canvas'),
      feature: document.createElement('canvas') };
    this.hasRef = false;
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
   * The reference photo and its 478 points. Also keeps a blurred copy: in
   * live-lighting mode the shader divides it out to get the reference detail.
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
    const low = this.target('refLow', W / 4, H / 4);
    const tmp = this.target('refTmp', W / 4, H / 4);
    this.copy(this.refTex, low);
    this.blur(low, tmp, faceWidth(points, W, H) / 16 / 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.hasRef = true;
  }

  clearReference() { this.hasRef = false; }

  /**
   * One frame.
   * f: { mirror, points (478 live points or null), gain, s (settings) }
   * s: { regions[], blend, lighting ('live'|'photo'), smooth, eyeSize,
   *      eyeColor [r,g,b], eyeAmt, lipColor [r,g,b], lipAmt,
   *      warmth, brightness, contrast, saturation, showMesh }
   */
  draw(video, f) {
    const gl = this.gl;
    const s = f.s;
    const W = video.videoWidth, H = video.videoHeight;
    gl.disable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    const pts = f.points;
    const transfer = this.hasRef && s.blend > 0 && s.regions.length > 0;
    let irisA = [0, 0, 0], irisB = [0, 0, 0];
    if (pts) {
      const d = displace(pts, s.eyeSize, W, H, this.disp);
      for (let i = 0; i < MESH_POINTS * 2; i++) this.lpos[i] = i % 2 ? pts[i >> 1].y : pts[i >> 1].x;
      const fw = faceWidth(pts, W, H);
      const Pq = i => [d[i * 2] * W / 4, d[i * 2 + 1] * H / 4];
      const Ph = i => [d[i * 2] * W / 2, d[i * 2 + 1] * H / 2];

      if (transfer && s.lighting === 'live') {
        const low = this.target('liveLow', W / 4, H / 4);
        this.copy(this.videoTex, low);
        this.blur(low, this.target('tmpQ', W / 4, H / 4), fw / 16 / 4);
      }
      if (s.smooth > 0) {
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
      if (s.lipAmt > 0 || s.eyeAmt > 0) {
        const c = this.canvases.feature;
        drawFeatureMask(c, Math.round(W / 2), Math.round(H / 2), Ph);
        gl.bindTexture(gl.TEXTURE_2D, this.featureTex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
      }
      irisA = iris(d, 468, W, H);
      irisB = iris(d, 473, W, H);
    }

    // Background: the whole camera frame, graded.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    const grade = [s.warmth, s.brightness, s.contrast, s.saturation];
    this.useProgram(this.bg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.uniform1i(this.u(this.bg, 'tex'), 0);
    gl.uniform1f(this.u(this.bg, 'mirror'), f.mirror ? 1 : 0);
    gl.uniform4fv(this.u(this.bg, 'grade'), grade);
    this.attrib(this.bg, 'p', this.quad, 2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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
    const T = this.targets;
    const units = [
      ['video', this.videoTex], ['liveLow', T.liveLow?.tex], ['liveSmooth', T.liveSmooth?.tex],
      ['ref', this.refTex], ['refLow', T.refLow?.tex], ['maskT', T.maskT?.tex],
      ['maskS', T.maskS?.tex], ['maskX', this.featureTex],
    ];
    units.forEach(([name, tex], i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, tex || null);
      gl.uniform1i(this.u(p, name), i);
    });
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(this.u(p, 'mirror'), f.mirror ? 1 : 0);
    gl.uniform2f(this.u(p, 'vsize'), W, H);
    gl.uniform1f(this.u(p, 'hasRef'), transfer ? 1 : 0);
    gl.uniform1f(this.u(p, 'blend'), s.blend);
    gl.uniform1f(this.u(p, 'lighting'), s.lighting === 'live' ? 1 : 0);
    gl.uniform1f(this.u(p, 'smoothAmt'), s.smooth);
    gl.uniform1f(this.u(p, 'lipAmt'), s.lipAmt);
    gl.uniform1f(this.u(p, 'eyeAmt'), s.eyeAmt);
    gl.uniform3fv(this.u(p, 'gain'), f.gain);
    gl.uniform3fv(this.u(p, 'lipColor'), s.lipColor);
    gl.uniform3fv(this.u(p, 'eyeColor'), s.eyeColor);
    gl.uniform3fv(this.u(p, 'iris0'), irisA);
    gl.uniform3fv(this.u(p, 'iris1'), irisB);
    gl.uniform4fv(this.u(p, 'grade'), grade);
    this.attrib(p, 'pos', this.posBuf, 2);
    this.attrib(p, 'lpos', this.lposBuf, 2);
    this.attrib(p, 'tc', this.tcBuf, 2);
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
