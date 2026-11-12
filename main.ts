// roblox-up-and-coming-forecast-01 entry (Node 22+, standard library only).
//
// Model: per-game rank trajectory with a time-aware blend of persistence and
// momentum, averaged over the 72 hourly slots of the scored window.
//
//   - Each game seen in the last 7 days of history gets a velocity v: the
//     exponentially-weighted mean of its recent rank changes per hour (last
//     <=3 intervals, halflife 24 h). Games that left the chart are projected
//     forward from their last rank by v * (hours since last seen).
//   - Momentum trajectory: mu_m(t) = r0 + v * TAU * (1 - exp(-t/TAU)) with
//     TAU = 12 h (velocity decays; a stalled climber stops climbing), and
//     sigma(t) = sqrt(S0^2 + D2 * t) with S0 = 2.5 ranks, D2 = 0.05 ranks^2/h.
//   - Persistence trajectory: the same diffusion from the last rank with no
//     velocity, but a tighter start sigma_p(t) = sqrt(0.8^2 + D2 * t).
//   - At hour t the forecast is a(t) * P_persist(t) + (1 - a(t)) * P_mom(t)
//     with a(t) = exp(-t/12): the first hours trust the chart as it stands
//     (concentrated), later hours trust the velocity trajectory (broad).
//     Place probabilities are the Normal rank distribution discretized to
//     places 1..10, time-averaged over the 72 hourly points.
//   - Temperature gamma = 1.3 sharpens the time-average (flattening was
//     severely punished in backtests); rows with p < 0.002 are dropped;
//     per-place sums are capped at 1 (proportional scale-down).
//   - Age-out: the Up-and-Coming chart only holds games published within
//     the last 28 days, so a game gets zero probability after its 28th day.
//     The birth date is details.created when the read carries it (the house
//     recorder's reads do), else the game's first sighting in the history.
//   - top10 is the greedy argmax assignment, one distinct game per place.
//
// Tuning (walk-forward splits A/B on the practice archive + the practice
// window, scored with the exact RULES formula): practice log score -39.711
// vs the house persistence baseline -43.512 and pure momentum -41.627;
// walk-forward mean -42.781 vs baseline -54.973. Full variant table and
// scenario analysis in README.md.
//
// Deterministic: no wall clock, no randomness, no network. Reads
// /input/history.ndjson + /input/task.json (or MAKEFUN_INPUT_DIR /
// MAKEFUN_OUTPUT_DIR locally) and writes /output/forecast.json.
// Output is rounded to 1e-6 per row, so repeated runs are byte-identical.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- tuned parameters (see README.md) ---
const TAU = 12.0; // velocity decay, hours
const D2 = 0.05; // diffusion variance rate, ranks^2 per hour
const S0 = 2.5; // momentum start sigma, ranks
const S0_PERSIST = 0.8; // persistence start sigma, ranks
const VEL_HALFLIFE = 24.0; // velocity recency halflife, hours
const VEL_INTERVALS = 3; // intervals used for velocity
const BLEND_H = 12.0; // persistence weight halflife, hours
const GAMMA = 1.3; // sharpening exponent
const MIN_P = 0.002; // drop rows below this
const HOURS = 72; // scored window, hourly slots
const CANDIDATE_DAYS = 7; // games seen within this many days of the close
const AGE_OUT_DAYS = 28; // Up-and-Coming chart age rule

interface ChartRow {
  rank: number;
  universe_id: number;
}

interface Snapshot {
  at: number; // ms since epoch
  chart: ChartRow[];
  createdMs: Map<number, number>; // universe_id -> created ms (when known)
}

interface Obs {
  t: number; // hours from the last snapshot (<= 0)
  rank: number;
}

/** Normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation (|err| < 1.5e-7). */
function normCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function erf(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

function parseSnapshots(text: string): Snapshot[] {
  const snaps: Snapshot[] = [];
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    let d: any;
    try {
      d = JSON.parse(line);
    } catch {
      continue;
    }
    if (!d || d.ok === false || !Array.isArray(d.up_and_coming)) continue;
    const at = Date.parse(d.at as string);
    if (Number.isNaN(at)) continue;
    const chart: ChartRow[] = [...d.up_and_coming]
      .sort((a: any, b: any) => a.rank - b.rank)
      .map((r: any) => ({ rank: r.rank as number, universe_id: r.universe_id as number }));
    const createdMs = new Map<number, number>();
    const det = d.details;
    if (det && typeof det === 'object') {
      const entries: Array<[string, any]> = Array.isArray(det)
        ? det.map((e: any) => [String(e.universe_id ?? e.universeId ?? ''), e] as [string, any])
        : Object.entries(det);
      for (const [key, val] of entries) {
        if (!val || typeof val.created !== 'string') continue;
        const ms = Date.parse(val.created);
        if (Number.isNaN(ms)) continue;
        const uid = Number(key);
        if (!Number.isNaN(uid)) createdMs.set(uid, ms);
      }
    }
    snaps.push({ at, chart, createdMs });
  }
  snaps.sort((a, b) => a.at - b.at);
  return snaps;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number;
  output: string;
};
const snaps = parseSnapshots(readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8'));
if (snaps.length === 0) throw new Error('no usable snapshots in history.ndjson');
const places = task.places || 10;
const tLast = snaps[snaps.length - 1]!.at;

// Observations per game: (hours from the last snapshot, rank), oldest first.
const obs = new Map<number, Obs[]>();
const createdMs = new Map<number, number>(); // latest known created date per game
const firstSeenMs = new Map<number, number>();
for (const s of snaps) {
  const t = (s.at - tLast) / 3600000;
  for (const r of s.chart) {
    const list = obs.get(r.universe_id) ?? [];
    list.push({ t, rank: r.rank });
    obs.set(r.universe_id, list);
    if (!firstSeenMs.has(r.universe_id)) firstSeenMs.set(r.universe_id, s.at);
  }
  for (const [uid, ms] of s.createdMs) createdMs.set(uid, ms);
}

// Candidate games: seen in the last CANDIDATE_DAYS days, with velocity.
interface Cand {
  r0: number;
  v: number;
  birthH: number; // hours from the last snapshot of the game's birth (<= ~0)
}
const cand = new Map<number, Cand>();
for (const [uid, o] of obs) {
  const last = o[o.length - 1]!;
  if (last.t < -24 * CANDIDATE_DAYS) continue;
  let v = 0;
  let wsum = 0;
  const ivs: Array<[Obs, Obs]> = [];
  for (let i = o.length - 1; i >= 1 && ivs.length < VEL_INTERVALS; i--) {
    ivs.push([o[i - 1]!, o[i]!]);
  }
  for (const [o0, o1] of ivs) {
    const dt = o1.t - o0.t;
    if (dt <= 0) continue;
    const w = Math.pow(0.5, (0 - o1.t) / VEL_HALFLIFE);
    v += w * ((o1.rank - o0.rank) / dt);
    wsum += w;
  }
  if (wsum > 0) v /= wsum;
  // If the game left the chart, project from its last rank forward.
  const r0 = last.rank + (last.t < 0 ? v * -last.t : 0);
  const created = createdMs.get(uid);
  const birthH =
    created !== undefined
      ? (created - tLast) / 3600000
      : (firstSeenMs.get(uid)! - tLast) / 3600000;
  cand.set(uid, { r0, v, birthH });
}

// Time-averaged blended place probabilities over the 72 hourly points.
const acc = new Map<string, number>(); // "uid|place" -> prob mass
for (const [uid, c] of cand) {
  for (let h = 0; h < HOURS; h++) {
    const t = h + 0.5;
    // Age-out: no probability once the game is older than 28 days.
    if (t - c.birthH > AGE_OUT_DAYS * 24) break;
    const a = Math.exp(-t / BLEND_H);
    const sig = Math.sqrt(S0 * S0 + D2 * t);
    const sigP = Math.sqrt(S0_PERSIST * S0_PERSIST + D2 * t);
    const muM = c.r0 + c.v * TAU * (1 - Math.exp(-t / TAU));
    for (let k = 1; k <= places; k++) {
      const pm = normCdf((k + 0.5 - muM) / sig) - normCdf((k - 0.5 - muM) / sig);
      const pp = normCdf((k + 0.5 - c.r0) / sigP) - normCdf((k - 0.5 - c.r0) / sigP);
      const p = a * pp + (1 - a) * pm;
      if (p > 1e-6) {
        const key = `${uid}|${k}`;
        acc.set(key, (acc.get(key) ?? 0) + p / HOURS);
      }
    }
  }
}

// Sharpen, cap each place's sum at 1, drop tiny rows.
if (GAMMA !== 1) {
  for (const [key, p] of acc) acc.set(key, Math.pow(p, GAMMA));
}
const perPlace = new Map<number, number>();
for (const [key, p] of acc) {
  const k = Number(key.split('|')[1]);
  perPlace.set(k, (perPlace.get(k) ?? 0) + p);
}
interface ProbRow {
  universe_id: number;
  place: number;
  p: number;
}
const probabilities: ProbRow[] = [];
for (const [key, p0] of acc) {
  const [uidStr, kStr] = key.split('|');
  const k = Number(kStr);
  let p = p0;
  const s = perPlace.get(k) ?? 0;
  if (s > 1) p = p / s;
  if (p >= MIN_P) {
    probabilities.push({
      universe_id: Number(uidStr),
      place: k,
      p: Math.round(p * 1e6) / 1e6,
    });
  }
}
probabilities.sort((a, b) => a.place - b.place || b.p - a.p || a.universe_id - b.universe_id);

// top10: greedy argmax assignment, one distinct game per place.
const byPlace = new Map<number, ProbRow[]>();
for (const row of probabilities) {
  const list = byPlace.get(row.place) ?? [];
  list.push(row);
  byPlace.set(row.place, list);
}
const lastChart = [...snaps[snaps.length - 1]!.chart]
  .sort((a, b) => a.rank - b.rank)
  .map((r) => r.universe_id);
const used = new Set<number>();
const top10: number[] = [];
for (let k = 1; k <= places; k++) {
  let pick: number | undefined;
  for (const row of byPlace.get(k) ?? []) {
    if (!used.has(row.universe_id)) {
      pick = row.universe_id;
      break;
    }
  }
  if (pick === undefined) {
    for (const uid of lastChart) {
      if (!used.has(uid)) {
        pick = uid;
        break;
      }
    }
  }
  if (pick === undefined) break;
  used.add(pick);
  top10.push(pick);
}

writeFileSync(join(OUTPUT_DIR, task.output || 'forecast.json'), `${JSON.stringify({ probabilities, top10 })}\n`);
