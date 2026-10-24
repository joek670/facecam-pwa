/* "Describe a look": text in, a settings patch out.
 *
 * With INTERPRETER_URL set, the text goes to the Worker, which asks Claude to
 * fill in the settings schema. Only the typed text is sent; no image, no
 * frame. Without it, or when the Worker cannot be reached, a word list on the
 * device reads the text. Both results go through clean() before use.
 */
import { INTERPRETER_URL } from './config.js';
import { DEFAULTS, clean } from './settings.js';

// Longest names first, so "light blue" wins over "blue".
const COLORS = Object.entries({
  'light blue': '#7fb2e6', 'ice blue': '#a9d3f0', 'dark blue': '#244f8f', 'dark brown': '#3d2616',
  'hot pink': '#e0307a', 'deep red': '#8e1525', blue: '#3f7fd0', green: '#3f9a5a', emerald: '#1f8f5f',
  hazel: '#8e7a3c', brown: '#6b4426', amber: '#c98a1c', gold: '#d4a017', grey: '#8a9399', gray: '#8a9399',
  silver: '#b8c2c8', violet: '#7a4fc2', purple: '#7a3fa0', red: '#c0202d', pink: '#e0708f',
  coral: '#ef6f5a', nude: '#c58b78', berry: '#8e2045', plum: '#6e2a4e', wine: '#6d1a2a',
  orange: '#e07020', black: '#202020',
}).sort((a, b) => b[0].length - a[0].length);

const PHOTO = /\b(reference|photo|picture|pic|their|his|her|swap|copy|transfer|like the|makeup|make-up)\b/;

const REGION_WORDS = [
  [/\b(whole face|full face|face ?swap|entire face|face)\b/, 'face'],
  [/\b(eyebrows?|brows?)\b/, 'brows'],
  [/\b(eyes?)\b/, 'eyes'],
  [/\b(nose)\b/, 'nose'],
  [/\b(lips?|mouth)\b/, 'lips'],
];

/** How strong a clause asks for, 0..1, or null when it does not say. */
function intensity(t) {
  const n = t.match(/(\d{1,3})\s*%/);
  if (n) return Math.min(100, +n[1]) / 100;
  if (/\b(very|really|super|strong(ly)?|lots of|heavy|heavily|max(imum)?|extreme(ly)?|much)\b/.test(t)) return 0.85;
  if (/\b(slight(ly)?|a bit|a little|subtle|subtly|gentle|gently|touch of|hint of)\b/.test(t)) return 0.3;
  return null;
}

/** Pull a colour out of the clause; returns [hex|null, the clause without it]. */
function takeColor(t) {
  const hex = t.match(/#[0-9a-f]{6}\b/);
  if (hex) return [hex[0], t.replace(hex[0], ' ')];
  for (const [name, value] of COLORS) {
    const re = new RegExp(`\\b${name}\\b`);
    if (re.test(t)) return [value, t.replace(re, ' ')];
  }
  return [null, t];
}

/** The on-device reader. Returns { patch, unknown[] }. */
export function parseLocal(text) {
  const patch = {};
  const unknown = [];
  // Join phrases that contain a clause word before splitting on it.
  const all = text.toLowerCase().replace(/\bblack\s*(and|&|n)\s*white\b/g, 'b&w');

  if (/^\s*(reset|clear( all)?|no filters?|natural|start over|undo all)\s*\.?\s*$/.test(all)) {
    const { regions, blend, lighting, ...edits } = DEFAULTS;
    return { patch: { ...edits }, unknown };
  }

  // A comma-separated part that names the photo is a transfer request, and
  // every feature in it counts: "eyes and brows from my photo".
  const clauses = [];
  for (const part of all.split(/[,.;\n]/)) {
    const photo = PHOTO.test(part);
    if (photo) {
      if (/\b(swap|features?|transfer|their (face|eyes|nose|lips))\b/.test(part)) patch.refMode = 'features';
      else if (/\b(look|makeup|make-up|style|vibe|tone)\b/.test(part)) patch.refMode = 'look';
      const regions = REGION_WORDS.filter(([re]) => re.test(part)).map(([, r]) => r);
      if (regions.length) {
        patch.regions = /\b(no|without|remove)\b/.test(part) ? [] : regions;
        const amt = intensity(takeColor(part)[1]);
        if (amt !== null) patch.blend = amt;
      } else if (/\b(nothing|none|no|stop|remove|without)\b/.test(part)) {
        patch.regions = [];
      }
    }
    for (const raw of part.split(/\band\b|\bwith\b|\bplus\b|\bthen\b/)) clauses.push([raw, photo]);
  }

  for (const [raw, photoPart] of clauses) {
    let t = raw.trim();
    if (!t) continue;
    const off = /\b(no|without|remove|turn off)\b/.test(t);
    let color;
    [color, t] = takeColor(t);
    const amt = intensity(t);
    // In a photo part, a feature named without a colour is the transfer above.
    const fromPhoto = photoPart && !color;
    let hit = fromPhoto && (PHOTO.test(t) || REGION_WORDS.some(([re]) => re.test(t)));

    if (/\b(keep (my|the) (light|lighting)|live light(ing)?|natural light(ing)?|match (the )?light(ing)?)\b/.test(t)) {
      patch.lighting = 'live'; hit = true;
    }
    if (/\b(exact|photo|original) colou?rs?\b/.test(t)) { patch.lighting = 'photo'; hit = true; }

    if (!fromPhoto && /\b(eyes?|iris(es)?|eye colou?r)\b/.test(t) && (color || off) && !/\b(big|bigger|large|larger)\b/.test(t)) {
      if (off) patch.eyeAmt = 0;
      else { patch.eyeColor = color; patch.eyeAmt = amt ?? 0.6; }
      hit = true;
    }
    if (!fromPhoto && /\b(lips?|lipstick|lip colou?r|mouth)\b/.test(t)) {
      if (off) patch.lipAmt = 0;
      else { patch.lipColor = color || (patch.lipColor ?? '#c0202d'); patch.lipAmt = amt ?? 0.55; }
      hit = true;
    }
    if (/\b(smooth(er|ing)?|airbrush(ed)?|soft(er)? skin|clear skin|flawless|beauty|retouch)\b/.test(t)) {
      patch.smooth = off ? 0 : amt ?? 0.55; hit = true;
    }
    if (/\b(big|bigger|large|larger|anime|doll|wide) eyes?\b/.test(t) || /\beyes? (bigger|larger)\b/.test(t)) {
      patch.eyeSize = off ? 0 : amt ?? 0.5; hit = true;
    }
    if (/\b(warm|warmer|golden|sunset|summer)\b/.test(t)) { patch.warmth = off ? 0 : (amt ?? 0.5) * 0.8; hit = true; }
    if (/\b(cool|cooler|cold|colder|icy|winter)\b/.test(t)) { patch.warmth = off ? 0 : -(amt ?? 0.5) * 0.8; hit = true; }
    if (/\b(bright|brighter|lighter|brighten)\b/.test(t)) { patch.brightness = off ? 0 : (amt ?? 0.5) * 0.6; hit = true; }
    if (/\b(dark|darker|dim|moody|darken)\b/.test(t)) {
      patch.brightness = off ? 0 : -(amt ?? 0.5) * 0.5; hit = true;
      if (/\bmoody\b/.test(t)) patch.contrast = 0.35;
    }
    if (/\b(contrast|punchy|crisp|dramatic)\b/.test(t)) {
      patch.contrast = off || /\blow contrast\b/.test(t) ? (off ? 0 : -0.4) : (amt ?? 0.5) * 0.7; hit = true;
    }
    if (/\b(flat|faded|matte)\b/.test(t)) { patch.contrast = -(amt ?? 0.5) * 0.6; hit = true; }
    if (/\b(black and white|b&w|b\/w|mono(chrome)?|gr[ae]y ?scale|noir)\b/.test(t)) { patch.saturation = -1; hit = true; }
    else if (/\b(vivid|vibrant|saturated|colou?rful|pop)\b/.test(t)) { patch.saturation = (amt ?? 0.5) * 0.8; hit = true; }
    else if (/\b(muted|desaturated?|washed|pastel)\b/.test(t)) { patch.saturation = -(amt ?? 0.5) * 0.7; hit = true; }
    if (/\b(show|hide)\b.*\bmesh\b/.test(t)) { patch.showMesh = /\bshow\b/.test(t); hit = true; }

    if (!hit) unknown.push(raw.trim());
  }
  return { patch, unknown };
}

/**
 * Read the text with Claude when a Worker is configured, else on the device.
 * Returns { patch, unknown[], via: 'claude' | 'device', error? }.
 */
export async function interpret(text, current) {
  if (INTERPRETER_URL && navigator.onLine !== false) {
    try {
      const res = await fetch(`${INTERPRETER_URL.replace(/\/$/, '')}/interpret`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ text, current }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = await res.json();
      return { patch: clean(body.patch), unknown: Array.isArray(body.unknown) ? body.unknown.map(String) : [], via: 'claude' };
    } catch (err) {
      const local = parseLocal(text);
      return { patch: clean(local.patch), unknown: local.unknown, via: 'device', error: String(err.message || err) };
    }
  }
  const local = parseLocal(text);
  return { patch: clean(local.patch), unknown: local.unknown, via: 'device' };
}
