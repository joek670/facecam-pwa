/* The filter settings: defaults, limits, and plain-language descriptions.
 *
 * Every change goes through clean(), whether it came from a slider, the
 * on-device word list, or the Claude interpreter. Anything outside the
 * schema is dropped, so a bad interpreter answer can change nothing it
 * should not.
 */
import { REGIONS } from './face.js';

export const DEFAULTS = Object.freeze({
  // 'look': recolour the live face with the photo's makeup and colour.
  // 'features': a face swap. The live face takes the photo's proportions
  // and its colours at full strength; the photo's pixels are never drawn.
  refMode: 'look',
  regions: ['face'],
  blend: 0.85,
  // In a face swap, also give the hair the photo's hair colour.
  refHair: false,
  eyeColor: '#3f7fd0',
  eyeAmt: 0,
  lipColor: '#b8323f',
  lipAmt: 0,
  smooth: 0,
  toon: 0,
  eyeSize: 0,
  hairColor: '#a4532e',
  hairAmt: 0,
  // Blur, then a solid colour over it: bgAmt 1 hides the background entirely.
  bgBlur: 0,
  bgColor: '#3a6f8c',
  bgAmt: 0,
  warmth: 0,
  brightness: 0,
  contrast: 0,
  saturation: 0,
  showMesh: false,
});

// Number settings: [min, max]. Amounts are 0..1, grades -1..1 with 0 = no change.
export const RANGES = {
  blend: [0, 1], eyeAmt: [0, 1], lipAmt: [0, 1], smooth: [0, 1], eyeSize: [0, 1], toon: [0, 1],
  hairAmt: [0, 1], bgBlur: [0, 1], bgAmt: [0, 1],
  warmth: [-1, 1], brightness: [-1, 1], contrast: [-1, 1], saturation: [-1, 1],
};

export const LABELS = {
  regions: 'From the photo', blend: 'Blend', refHair: "The photo's hair colour", eyeColor: 'Eye colour',
  eyeAmt: 'Eye colour', lipColor: 'Lip colour', lipAmt: 'Lip colour', smooth: 'Smooth skin',
  eyeSize: 'Eye size', toon: 'Cartoon style', hairColor: 'Hair colour', hairAmt: 'Hair colour',
  bgBlur: 'Background blur', bgColor: 'Background colour', bgAmt: 'Background colour',
  warmth: 'Warmth', brightness: 'Brightness', contrast: 'Contrast',
  saturation: 'Saturation', showMesh: 'Tracking mesh',
};

const COLOR_KEYS = ['eyeColor', 'lipColor', 'hairColor', 'bgColor'];

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
    } else if (COLOR_KEYS.includes(k) && typeof v === 'string' && HEX.test(v)) {
      out[k] = v.toLowerCase();
    } else if (k === 'refMode' && (v === 'look' || v === 'features')) {
      out[k] = v;
    } else if ((k === 'showMesh' || k === 'refHair') && typeof v === 'boolean') {
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
  if ('refMode' in p) lines.push(p.refMode === 'look' ? "The photo's look (colour and makeup)" : "Face swap: the photo's shape and colours");
  if ('regions' in p) lines.push(p.regions.length
    ? `From the photo: ${p.regions.map(r => REGION_LABELS[r].toLowerCase()).join(', ')}` : 'Nothing from the photo');
  if ('blend' in p) lines.push(`Blend ${pct(p.blend)}`);
  if ('refHair' in p) lines.push(p.refHair ? "The photo's hair colour (in Face swap)" : 'Own hair colour');
  const colour = (amt, col, label) => {
    if (!(amt in p || col in p)) return;
    lines.push(p[amt] === 0 ? `${label} off`
      : `${label} ${p[col] || ''} ${amt in p ? pct(p[amt]) : ''}`.replace(/\s+/g, ' ').trim());
  };
  colour('eyeAmt', 'eyeColor', 'Eye colour');
  colour('lipAmt', 'lipColor', 'Lip colour');
  colour('hairAmt', 'hairColor', 'Hair colour');
  for (const k of ['smooth', 'eyeSize', 'toon', 'bgBlur']) if (k in p) lines.push(`${LABELS[k]} ${pct(p[k])}`);
  colour('bgAmt', 'bgColor', 'Background colour');
  for (const k of ['warmth', 'brightness', 'contrast', 'saturation']) if (k in p) lines.push(`${LABELS[k]} ${signed(p[k])}`);
  if ('showMesh' in p) lines.push(p.showMesh ? 'Mesh on' : 'Mesh off');
  return lines;
}

/** '#rrggbb' to [r, g, b] in 0..1, for the shader. */
export function rgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}
