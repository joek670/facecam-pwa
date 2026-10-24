/* Face Cam — camera, reference photo, settings, capture and the render loop.
 * face.js holds the tracking and the rendering; settings.js the schema;
 * interpret.js turns "Describe a look" into settings. */
import { createLandmarker, buildMesh, meanColor, Renderer, REGIONS } from './face.js';
import { DEFAULTS, REGION_LABELS, clean, describe, rgb } from './settings.js';
import { interpret } from './interpret.js';

const $ = id => document.getElementById(id);
const els = {
  sub: $('sub'), stage: $('stage'), out: $('out'), cam: $('cam'), startbox: $('startbox'),
  start: $('start'), status: $('status'), snap: $('snap'), flip: $('flip'),
  refpick: $('refpick'), refthumb: $('refthumb'), reflabel: $('reflabel'), clearref: $('clearref'),
  notice: $('notice'), shots: $('shots'), noshots: $('noshots'),
  lookform: $('lookform'), look: $('look'), apply: $('apply'), via: $('via'), said: $('said'),
  regions: $('regions'), lighting: $('lighting'), lightingrow: $('lightingrow'), refMode: $('refMode'), eyeColor: $('eyeColor'), lipColor: $('lipColor'),
  showMesh: $('showMesh'), reset: $('reset'),
};

const state = {
  videoLm: null, imageLm: null, renderer: null,
  stream: null, facing: 'user', running: false,
  livePoints: null, lastVideoTime: -1, frame: 0,
  refPoints: null, refMean: null, liveMean: null, gain: [1, 1, 1],
};
const scratch = document.createElement('canvas');

function notice(msg, isErr = false) {
  els.notice.textContent = msg || '';
  els.notice.classList.toggle('hide', !msg);
  els.notice.classList.toggle('err', isErr);
}

// ---------------------------------------------------------------------------
// Settings. One object; every control and the text box write to it through
// setSettings(), which validates, redraws the controls and saves.
// ---------------------------------------------------------------------------

const SETTINGS_KEY = 'facecam-settings';
let settings = { ...DEFAULTS };
try {
  const saved = clean(JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'));
  // Saved before Look mode existed: those regions were picked for a face
  // swap, so start Look mode from its own default instead.
  if (!('refMode' in saved)) delete saved.regions;
  settings = { ...DEFAULTS, ...saved };
} catch (e) {}

// [key, label, container, signed]. Signed sliders run -100..100 around 0.
const SLIDERS = [
  ['blend', 'Blend', 'refsliders', false],
  ['eyeAmt', 'Eye colour amount', 'editsliders', false],
  ['eyeSize', 'Eye size', 'editsliders', false],
  ['smooth', 'Smooth skin', 'editsliders', false],
  ['lipAmt', 'Lip colour amount', 'lipsliders', false],
  ['warmth', 'Warmth', 'gradesliders', true],
  ['brightness', 'Brightness', 'gradesliders', true],
  ['contrast', 'Contrast', 'gradesliders', true],
  ['saturation', 'Saturation', 'gradesliders', true],
];
const sliders = {};

function buildControls() {
  for (const [key, label, box, signed] of SLIDERS) {
    const row = document.createElement('label');
    const input = document.createElement('input');
    input.type = 'range';
    input.min = signed ? -100 : 0;
    input.max = 100;
    input.setAttribute('aria-label', label);
    const val = document.createElement('span');
    val.className = 'val';
    row.append(label + ' ', input, val);
    input.addEventListener('input', () => setSettings({ [key]: input.value / 100 }, { fromControl: key }));
    $(box).append(row);
    sliders[key] = { input, val, signed };
  }
  for (const r of REGIONS) {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = REGION_LABELS[r];
    b.dataset.region = r;
    b.addEventListener('click', () => {
      const on = settings.regions.includes(r);
      setSettings({ regions: on ? settings.regions.filter(x => x !== r) : [...settings.regions, r] });
    });
    els.regions.append(b);
  }
  els.lighting.addEventListener('change', () => setSettings({ lighting: els.lighting.value }));
  els.refMode.addEventListener('change', () => setSettings({ refMode: els.refMode.value }));
  els.eyeColor.addEventListener('input', () => setSettings({ eyeColor: els.eyeColor.value,
    ...(settings.eyeAmt === 0 ? { eyeAmt: 0.6 } : {}) }));
  els.lipColor.addEventListener('input', () => setSettings({ lipColor: els.lipColor.value,
    ...(settings.lipAmt === 0 ? { lipAmt: 0.55 } : {}) }));
  els.showMesh.addEventListener('change', () => setSettings({ showMesh: els.showMesh.checked }));
  els.reset.addEventListener('click', () => {
    const { regions, blend, lighting, ...edits } = DEFAULTS;
    setSettings(edits);
  });
}

function syncControls(skip) {
  for (const [key, { input, val, signed }] of Object.entries(sliders)) {
    const v = Math.round(settings[key] * 100);
    if (key !== skip) input.value = v;
    val.textContent = signed ? (v > 0 ? `+${v}` : `${v}`) : `${v}%`;
  }
  for (const b of els.regions.children) b.setAttribute('aria-pressed', settings.regions.includes(b.dataset.region));
  els.lighting.value = settings.lighting;
  els.refMode.value = settings.refMode;
  // Live lighting vs photo colours only means something for a face swap.
  els.lightingrow.classList.toggle('hide', settings.refMode !== 'features');
  els.eyeColor.value = settings.eyeColor;
  els.lipColor.value = settings.lipColor;
  els.showMesh.checked = settings.showMesh;
}

function setSettings(patch, { fromControl } = {}) {
  settings = { ...settings, ...clean(patch) };
  syncControls(fromControl);
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch (e) {}
}

/** The settings in the form the renderer takes. */
function renderSettings() {
  return { ...settings, eyeColor: rgb(settings.eyeColor), lipColor: rgb(settings.lipColor) };
}

// ---------------------------------------------------------------------------
// Describe a look
// ---------------------------------------------------------------------------

async function applyLook(e) {
  e.preventDefault();
  const text = els.look.value.trim();
  if (!text) return;
  els.apply.disabled = true;
  els.via.textContent = 'Reading…';
  try {
    await readLook(text);
  } catch (err) {
    // Show the real cause: a silent failure here looks like "nothing works".
    console.error(err);
    els.via.textContent = '';
    els.said.textContent = `Could not apply that: ${err?.message || err}`;
    els.said.classList.remove('hide');
  } finally {
    els.apply.disabled = false;
  }
}

async function readLook(text) {
  const { patch, unknown, via, error } = await interpret(text, settings);
  els.via.textContent = via === 'claude' ? 'Read by Claude' : error
    ? 'Interpreter unreachable; read on this device' : 'Read on this device';
  setSettings(patch);

  const lines = describe(patch);
  els.said.textContent = '';
  els.said.append(lines.length ? `Understood: ${lines.join(' · ')}` : 'Nothing in that matched a setting.');
  if (unknown.length) {
    const s = document.createElement('span');
    s.textContent = `Not understood: ${unknown.join('; ')}`;
    els.said.append(s);
  }
  if (patch.regions?.length && !state.refPoints) {
    const s = document.createElement('span');
    s.textContent = 'Add a reference photo for the features to come from.';
    els.said.append(s);
  }
  els.said.classList.remove('hide');
}

// ---------------------------------------------------------------------------
// Reference photo store. One blob in IndexedDB so the photo survives a
// relaunch. A failure here costs only that: the photo works for this session.
// ---------------------------------------------------------------------------

function idb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open('facecam', 1);
    req.onupgradeneeded = () => req.result.createObjectStore('kv');
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}
async function kv(mode, fn) {
  const db = await idb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('kv', mode);
    const req = fn(tx.objectStore('kv'));
    tx.oncomplete = () => resolve(req && req.result);
    tx.onerror = () => reject(tx.error);
  });
}
const store = {
  get: k => kv('readonly', s => s.get(k)).catch(() => null),
  set: (k, v) => kv('readwrite', s => s.put(v, k)).catch(() => {}),
  del: k => kv('readwrite', s => s.delete(k)).catch(() => {}),
};

// ---------------------------------------------------------------------------
// Reference photo
// ---------------------------------------------------------------------------

const REF_MAX = 1024;

/** Decode, honour EXIF rotation, and scale down to REF_MAX on the long side. */
async function decodeImage(blob) {
  const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const scale = Math.min(1, REF_MAX / Math.max(bmp.width, bmp.height));
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * scale);
  c.height = Math.round(bmp.height * scale);
  c.getContext('2d').drawImage(bmp, 0, 0, c.width, c.height);
  bmp.close?.();
  return c;
}

async function setReference(blob, { save }) {
  if (!state.imageLm) return;
  let img;
  try {
    img = await decodeImage(blob);
  } catch (err) {
    notice('That file could not be opened as an image.', true);
    return;
  }
  const result = state.imageLm.detect(img);
  const points = result.faceLandmarks?.[0];
  if (!points) {
    notice('No face found in that photo. Use a clear, front-facing photo with the whole face in frame.', true);
    return;
  }
  state.refPoints = points;
  state.refMean = meanColor(img, points, img.width, img.height, scratch);
  state.renderer.setReference(img, points);
  els.refthumb.src = img.toDataURL('image/jpeg', 0.8);
  els.refthumb.classList.remove('hide');
  els.reflabel.textContent = 'Change';
  els.clearref.disabled = false;
  notice('');
  if (save) await store.set('ref', blob);
}

async function clearReference() {
  state.refPoints = null;
  state.refMean = null;
  state.renderer?.clearReference();
  els.refthumb.classList.add('hide');
  els.refthumb.removeAttribute('src');
  els.reflabel.textContent = 'Reference';
  els.clearref.disabled = true;
  await store.del('ref');
}

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    notice('This browser gives no camera access here. The camera needs HTTPS, or localhost for development.', true);
    return;
  }
  stopCamera();
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    const denied = err.name === 'NotAllowedError' || err.name === 'SecurityError';
    notice(denied
      ? 'Camera access was refused. Allow the camera for this site in the browser settings, then try again.'
      : `The camera could not start (${err.name}).`, true);
    return;
  }
  els.cam.srcObject = state.stream;
  await els.cam.play();
  state.lastVideoTime = -1;
  state.livePoints = null;
  els.startbox.classList.add('hide');
  els.status.classList.remove('hide');
  els.snap.disabled = false;
  notice('');

  // Only offer Flip when there is a second camera to flip to. Device labels
  // and counts are only reliable after permission is granted.
  try {
    const cams = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
    els.flip.disabled = cams.length < 2;
  } catch (e) {}

  if (!state.running) {
    state.running = true;
    requestAnimationFrame(loop);
  }
}

function stopCamera() {
  state.stream?.getTracks().forEach(t => t.stop());
  state.stream = null;
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

function loop() {
  const v = els.cam;
  if (v.readyState >= 2 && v.videoWidth) {
    if (els.out.width !== v.videoWidth || els.out.height !== v.videoHeight) {
      els.out.width = v.videoWidth;
      els.out.height = v.videoHeight;
    }
    // Track once per new camera frame, not once per display frame.
    if (v.currentTime !== state.lastVideoTime) {
      state.lastVideoTime = v.currentTime;
      const r = state.videoLm.detectForVideo(v, performance.now());
      state.livePoints = r.faceLandmarks?.[0] || null;
    }

    // Skin tone: move the reference colours toward the live face's mean.
    // Photo colours: the whole colour, brightness included. Live lighting:
    // the hue only, because the shader already matches brightness per pixel.
    // Sampled every 12 frames and eased, so a passing shadow does not flicker.
    const pts = state.livePoints;
    if (pts && state.refMean && state.frame % 12 === 0) {
      state.liveMean = meanColor(v, pts, v.videoWidth, v.videoHeight, scratch) || state.liveMean;
    }
    let target = [1, 1, 1];
    let lumNorm = 1;
    if (state.refMean && state.liveMean) {
      const lum = c => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
      // Look mode: how much brighter the photo's face is overall, so only
      // local darkening (makeup, contour) transfers, not the photo's exposure.
      lumNorm = clamp(lum(state.refMean) / Math.max(1, lum(state.liveMean)), 0.3, 3);
      const ratio = state.liveMean.map((l, i) => l / Math.max(1, state.refMean[i]));
      if (settings.lighting === 'photo') target = ratio.map(r => clamp(r, 0.6, 1.6));
      else target = ratio.map(r => clamp(r * lumNorm, 0.8, 1.25));
    }
    state.gain = state.gain.map((g, i) => g + (target[i] - g) * 0.15);

    state.renderer.draw(v, {
      mirror: state.facing === 'user',
      points: pts,
      gain: state.gain,
      lumNorm,
      s: renderSettings(),
    });
    const areas = settings.regions.map(r => REGION_LABELS[r].toLowerCase()).join(', ');
    els.status.textContent = !pts ? 'Looking for a face…'
      : !state.refPoints ? 'Face found.'
      : !settings.regions.length ? 'Face found. No areas chosen from the photo.'
      : settings.refMode === 'look' ? `The photo's look on: ${areas}.`
      : `The photo's features: ${areas}.`;
    state.frame++;
  }
  if (state.stream) requestAnimationFrame(loop);
  else state.running = false;
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

function capture() {
  els.out.toBlob(blob => {
    if (!blob) return;
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const file = new File([blob], `facecam-${stamp}.jpg`, { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const b = document.createElement('button');
    b.setAttribute('aria-label', `Save or share ${file.name}`);
    const img = document.createElement('img');
    img.src = url;
    img.alt = '';
    b.append(img);
    b.addEventListener('click', () => saveShot(file, url));
    els.shots.prepend(b);
    els.noshots.classList.add('hide');
    els.stage.classList.remove('flash');
    void els.stage.offsetWidth;
    els.stage.classList.add('flash');
  }, 'image/jpeg', 0.92);
}

/** Share sheet where the browser has one (iOS: Save Image), a download otherwise. */
async function saveShot(file, url) {
  if (navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file] });
      return;
    } catch (err) {
      if (err.name === 'AbortError') return;
    }
  }
  const a = document.createElement('a');
  a.href = url;
  a.download = file.name;
  a.click();
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

function wire() {
  buildControls();
  syncControls();
  els.lookform.addEventListener('submit', applyLook);
  els.start.addEventListener('click', startCamera);
  els.snap.addEventListener('click', capture);
  els.flip.addEventListener('click', () => {
    state.facing = state.facing === 'user' ? 'environment' : 'user';
    startCamera();
  });
  els.refpick.addEventListener('change', () => {
    const f = els.refpick.files?.[0];
    if (f) setReference(f, { save: true });
    els.refpick.value = '';
  });
  els.clearref.addEventListener('click', clearReference);
  // Release the camera when the app is backgrounded; iOS shows the camera
  // indicator otherwise, and the stream is dead on return anyway.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && state.stream) {
      stopCamera();
      els.startbox.classList.remove('hide');
      els.status.classList.add('hide');
      els.snap.disabled = true;
    }
  });
}

async function init() {
  wire();
  try {
    [state.videoLm, state.imageLm] = await Promise.all([createLandmarker('VIDEO'), createLandmarker('IMAGE')]);
    state.renderer = new Renderer(els.out, buildMesh());
  } catch (err) {
    console.error(err);
    els.sub.textContent = 'Face model failed to load';
    notice('The face model could not load. Check the connection and reload; after one good load it works offline.', true);
    return;
  }
  els.sub.textContent = 'On-device face filter';
  els.start.disabled = false;
  const saved = await store.get('ref');
  if (saved) await setReference(saved, { save: false });
}

if ('serviceWorker' in navigator && (location.protocol === 'https:' || /^(localhost|127\.0\.0\.1)$/.test(location.hostname))) {
  navigator.serviceWorker.register('sw.js').catch(() => {});
}
init();
