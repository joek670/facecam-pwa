/* Face Cam — camera, reference photo, capture and the render loop.
 * face.js holds the tracking and the warp; this file is the app around it. */
import { createLandmarker, buildMesh, meanColor, Renderer } from './face.js';

const $ = id => document.getElementById(id);
const els = {
  sub: $('sub'), stage: $('stage'), out: $('out'), cam: $('cam'), startbox: $('startbox'),
  start: $('start'), status: $('status'), snap: $('snap'), flip: $('flip'),
  refpick: $('refpick'), refthumb: $('refthumb'), reflabel: $('reflabel'),
  strength: $('strength'), strengthval: $('strengthval'), tone: $('tone'), mesh: $('mesh'),
  clearref: $('clearref'), notice: $('notice'), shots: $('shots'), noshots: $('noshots'),
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

    // Skin tone: scale the reference colours toward the live face's mean.
    // Sampled every 12 frames and eased, so a passing shadow does not flicker.
    const pts = state.livePoints;
    if (pts && state.refMean && state.frame % 12 === 0) {
      state.liveMean = meanColor(v, pts, v.videoWidth, v.videoHeight, scratch) || state.liveMean;
    }
    const target = els.tone.checked && state.refMean && state.liveMean
      ? state.liveMean.map((l, i) => clamp(l / Math.max(1, state.refMean[i]), 0.6, 1.6))
      : [1, 1, 1];
    state.gain = state.gain.map((g, i) => g + (target[i] - g) * 0.15);

    state.renderer.draw(v, {
      mirror: state.facing === 'user',
      points: pts,
      strength: els.strength.value / 100,
      gain: state.gain,
      showMesh: els.mesh.checked,
    });
    els.status.textContent = !pts ? 'Looking for a face…'
      : !state.refPoints ? 'Face found. Add a reference photo to apply the filter.'
      : 'Filter on.';
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
  const showStrength = () => (els.strengthval.textContent = `${els.strength.value}%`);
  els.strength.addEventListener('input', showStrength);
  showStrength();
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
