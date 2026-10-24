/* The filter settings: defaults, limits, and plain-language descriptions.
 *
 * Every change goes through clean(), whether it came from a slider, the
 * on-device word list, or the Claude interpreter. Anything outside the
 * schema is dropped, so a bad interpreter answer can change nothing it
 * should not.
 */
import { REGIONS } from './face.js';

export const DEFAULTS = Object.freeze({
  regions: ['eyes', 'brows'],
  blend: 0.85,
  lighting: 'live',
  eyeColor: '#3f7fd0',
  eyeAmt: 0,
  lipColor: '#b8323f',
  lipAmt: 0,
  smooth: 0,
  eyeSize: 0,
  warmth: 0,
  brightness: 0,
  contrast: 0,
  saturation: 0,
  showMesh: false,
});

// Number settings: [min, max]. Amounts are 0..1, grades -1..1 with 0 = no change.
export const RANGES = {
  blend: [0, 1], eyeAmt: [0, 1], lipAmt: [0, 1], smooth: [0, 1], eyeSize: [0, 1],
  warmth: [-1, 1], brightness: [-1, 1], contrast: [-1, 1], saturation: [-1, 1],
};

export const LABELS = {
  regions: 'From the photo', blend: 'Blend', lighting: 'Colour', eyeColor: 'Eye colour',
  eyeAmt: 'Eye colour', lipColor: 'Lip colour', lipAmt: 'Lip colour', smooth: 'Smooth skin',
  eyeSize: 'Eye size', warmth: 'Warmth', brightness: 'Brightness', contrast: 'Contrast',
  saturation: 'Saturation', showMesh: 'Tracking mesh',
};

export const REGION_LABELS = { face: 'Whole face', eyes: 'Eyes', brows: 'Brows', nose: 'Nose', lips: 'Lips' };

const HEX = /^#[0-9a-f]{6}$/i;

/** The valid part of a settings patch. */
export function clean(patch) {
  const out = {};
  if (!patch || typeof patch !== 'object') return out;
  for (const [k, v] of Object.entries(patch)) {
    if (k in RANGES && typeof v === 'number' && Number.isFinite(v)) {
      const [lo, hi] = RANGES[k];
      out[k] = Math.min(hi, Math.max(lo, v));
    } else if ((k === 'eyeColor' || k === 'lipColor') && typeof v === 'string' && HEX.test(v)) {
      out[k] = v.toLowerCase();
    } else if (k === 'lighting' && (v === 'live' || v === 'photo')) {
      out[k] = v;
    } else if (k === 'showMesh' && typeof v === 'boolean') {
      out[k] = v;
    } else if (k === 'regions' && Array.isArray(v)) {
      out[k] = REGIONS.filter(r => v.includes(r));
    }
  }
  return out;
}

const pct = v => `${Math.round(v * 100)}%`;
const signed = v => (v > 0 ? '+' : '') + Math.round(v * 100);

/** One short line per changed setting, for the "Understood" read-back. */
export function describe(patch) {
  const lines = [];
  const p = patch;
  if ('regions' in p) lines.push(p.regions.length
    ? `From the photo: ${p.regions.map(r => REGION_LABELS[r].toLowerCase()).join(', ')}` : 'Nothing from the photo');
  if ('blend' in p) lines.push(`Blend ${pct(p.blend)}`);
  if ('lighting' in p) lines.push(p.lighting === 'live' ? 'Live lighting' : 'Photo colours');
  if ('eyeAmt' in p || 'eyeColor' in p) lines.push(p.eyeAmt === 0 ? 'Eye colour off'
    : `Eye colour ${p.eyeColor || ''} ${'eyeAmt' in p ? pct(p.eyeAmt) : ''}`.replace(/\s+/g, ' ').trim());
  if ('lipAmt' in p || 'lipColor' in p) lines.push(p.lipAmt === 0 ? 'Lip colour off'
    : `Lip colour ${p.lipColor || ''} ${'lipAmt' in p ? pct(p.lipAmt) : ''}`.replace(/\s+/g, ' ').trim());
  for (const k of ['smooth', 'eyeSize']) if (k in p) lines.push(`${LABELS[k]} ${pct(p[k])}`);
  for (const k of ['warmth', 'brightness', 'contrast', 'saturation']) if (k in p) lines.push(`${LABELS[k]} ${signed(p[k])}`);
  if ('showMesh' in p) lines.push(p.showMesh ? 'Mesh on' : 'Mesh off');
  return lines;
}

/** '#rrggbb' to [r, g, b] in 0..1, for the shader. */
export function rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
