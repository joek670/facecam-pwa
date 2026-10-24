/* Face Cam — face tracking and the reference-photo warp.
 *
 * MediaPipe Face Landmarker gives 478 points per face: 468 mesh points plus
 * 10 iris points. The mesh points have the same meaning in every image —
 * point 1 is always the nose tip — so a reference photo and a live frame can
 * be matched point for point with no alignment step.
 *
 * The filter cuts the reference face into the mesh's triangles and draws each
 * one at the live face's position for that triangle. The texture coordinates
 * come from the reference photo, the positions from the camera, so the
 * reference face follows the live head, jaw and expression.
 *
 * Everything runs on the device. The model and WASM come from the CDN once and
 * the service worker keeps them; no frame or photo leaves the browser.
 */

export const TASKS_VERSION = '0.10.14';
const TASKS_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}`;
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

// The 468 mesh points. The iris points are not part of the tesselation.
const MESH_POINTS = 468;

let vision = null;
let FaceLandmarker = null;

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
  // The tesselation lists edges only. Each triangle is a 3-cycle of edges:
  // 854 of them, and 1322 edges - 468 points + 1 = 855 faces, so every cycle
  // is a real triangle and the one face left over is the mouth opening. The
  // mesh has no triangles across the lips, so an open live mouth shows the
  // live teeth, not stretched reference lips.
  const tris = [];
  for (let a = 0; a < MESH_POINTS; a++) {
    for (const b of adj[a]) {
      if (b <= a) continue;
      for (const c of adj[b]) {
        if (c <= b || !adj[a].has(c)) continue;
        tris.push(a, b, c);
      }
    }
  }
  const lines = [];
  for (const { start, end } of edges) lines.push(start, end);

  // Feather the edge: opacity rises over the three rings in from the face
  // outline, so the reference face fades into the live skin and has no seam.
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
// WebGL renderer
// ---------------------------------------------------------------------------

const VS_BG = `
attribute vec2 p;
uniform float mirror;
varying vec2 uv;
void main() {
  uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
  if (mirror > 0.5) uv.x = 1.0 - uv.x;
  gl_Position = vec4(p, 0.0, 1.0);
}`;
const FS_BG = `
precision mediump float;
uniform sampler2D tex;
varying vec2 uv;
void main() { gl_FragColor = texture2D(tex, uv); }`;

const VS_MESH = `
attribute vec2 pos;
attribute vec2 tc;
attribute float a;
uniform float mirror;
varying vec2 uv;
varying float va;
void main() {
  float x = mirror > 0.5 ? 1.0 - pos.x : pos.x;
  gl_Position = vec4(x * 2.0 - 1.0, 1.0 - pos.y * 2.0, 0.0, 1.0);
  uv = tc;
  va = a;
}`;
const FS_MESH = `
precision mediump float;
uniform sampler2D tex;
uniform vec3 gain;
uniform float strength;
varying vec2 uv;
varying float va;
void main() {
  vec4 c = texture2D(tex, uv);
  gl_FragColor = vec4(clamp(c.rgb * gain, 0.0, 1.0), va * strength);
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
    this.gl = gl;
    this.mesh = mesh;
    this.bg = program(gl, VS_BG, FS_BG);
    this.face = program(gl, VS_MESH, FS_MESH);
    this.line = program(gl, VS_MESH, FS_LINE);
    this.videoTex = texture(gl);
    this.refTex = texture(gl);

    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    this.posBuf = gl.createBuffer();
    this.tcBuf = gl.createBuffer();
    this.alphaBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.alphaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.alpha, gl.STATIC_DRAW);
    this.triBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.tris, gl.STATIC_DRAW);
    this.lineBuf = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.lines, gl.STATIC_DRAW);

    this.pos = new Float32Array(MESH_POINTS * 2);
    this.hasRef = false;
    this.enabled = new Set();
  }

  setReference(image, points) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.refTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
    const tc = new Float32Array(MESH_POINTS * 2);
    for (let i = 0; i < MESH_POINTS; i++) { tc[i * 2] = points[i].x; tc[i * 2 + 1] = points[i].y; }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.tcBuf);
    gl.bufferData(gl.ARRAY_BUFFER, tc, gl.STATIC_DRAW);
    this.hasRef = true;
  }

  clearReference() { this.hasRef = false; }

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

  /**
   * One frame: the camera image, then the reference face over the live
   * points, then the mesh lines if asked for.
   * opts: { mirror, points, strength, gain, showMesh }
   */
  draw(video, opts) {
    const gl = this.gl;
    const { width, height } = gl.canvas;
    gl.viewport(0, 0, width, height);
    gl.disable(gl.BLEND);

    this.useProgram(this.bg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.videoTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.uniform1i(gl.getUniformLocation(this.bg, 'tex'), 0);
    gl.uniform1f(gl.getUniformLocation(this.bg, 'mirror'), opts.mirror ? 1 : 0);
    this.attrib(this.bg, 'p', this.quad, 2);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    const pts = opts.points;
    if (!pts) return;
    for (let i = 0; i < MESH_POINTS; i++) { this.pos[i * 2] = pts[i].x; this.pos[i * 2 + 1] = pts[i].y; }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.pos, gl.DYNAMIC_DRAW);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    if (this.hasRef && opts.strength > 0) {
      const p = this.face;
      this.useProgram(p);
      gl.bindTexture(gl.TEXTURE_2D, this.refTex);
      gl.uniform1i(gl.getUniformLocation(p, 'tex'), 0);
      gl.uniform1f(gl.getUniformLocation(p, 'mirror'), opts.mirror ? 1 : 0);
      gl.uniform1f(gl.getUniformLocation(p, 'strength'), opts.strength);
      gl.uniform3fv(gl.getUniformLocation(p, 'gain'), opts.gain);
      this.attrib(p, 'pos', this.posBuf, 2);
      this.attrib(p, 'tc', this.tcBuf, 2);
      this.attrib(p, 'a', this.alphaBuf, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.triBuf);
      gl.drawElements(gl.TRIANGLES, this.mesh.tris.length, gl.UNSIGNED_SHORT, 0);
    }

    if (opts.showMesh) {
      const p = this.line;
      this.useProgram(p);
      gl.uniform1f(gl.getUniformLocation(p, 'mirror'), opts.mirror ? 1 : 0);
      gl.uniform4fv(gl.getUniformLocation(p, 'color'), [0.45, 0.9, 0.84, 0.55]);
      this.attrib(p, 'pos', this.posBuf, 2);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.lineBuf);
      gl.drawElements(gl.LINES, this.mesh.lines.length, gl.UNSIGNED_SHORT, 0);
    }
  }
}
