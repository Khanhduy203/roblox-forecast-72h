# roblox-up-and-coming-forecast-01 — entry

Forecasts the Roblox Up-and-Coming chart's top 10 at every hour of the 72
hours after entries close (2026-11-13T06:00Z → 2026-11-16T06:00Z). Node 22+,
standard library only, deterministic, no network. Entry point: `node main.ts`
(reads `/input/history.ndjson` + `/input/task.json`, writes
`/output/forecast.json`; `MAKEFUN_INPUT_DIR` / `MAKEFUN_OUTPUT_DIR` stand in
locally).

## Model

Per-game rank trajectory, time-averaged over the 72 hourly slots, as a
time-aware blend of persistence and momentum:

- **Candidates**: every game seen in the last 7 days of history (plus the last
  chart). Games that left the chart are projected forward from their last rank.
- **Velocity**: exponentially-weighted mean of the last ≤3 rank changes per
  hour (halflife 24 h). Log player-count growth was tested and carries no
  signal (correlation −0.020 with rank change); rank velocity persists
  (correlation 0.223). Player-count *level* is anti-predictive.
- **Momentum trajectory**: `mu(t) = r0 + v·12·(1 − exp(−t/12))`,
  `sigma(t) = sqrt(2.5² + 0.05·t)`. Velocity decays with a 12 h time constant,
  so a stalled climber stops climbing within about a day.
- **Persistence trajectory**: same diffusion from the last rank with no
  velocity and a tighter start sigma `sqrt(0.8² + 0.05·t)`.
- **Blend**: at hour `t`, `p(t) = exp(−t/12)·P_persist(t) + (1 − exp(−t/12))·P_mom(t)`.
  The first hours trust the chart as it stands (concentrated); later hours
  trust the velocity trajectory (broad). Place probabilities are the Normal
  rank distribution discretized to places 1–10.
- **Calibration**: temperature `gamma = 1.3` sharpens the time-average
  (flattening was severely punished in backtests); rows with `p < 0.002`
  dropped; per-place sums capped at 1 (proportional scale-down).
- **Age-out**: the chart only holds games published within the last 28 days,
  so a game gets zero probability after its 28th day. Birth date is
  `details.created` when the read carries it (house recorder reads do), else
  first sighting in the history.
- **top10**: greedy argmax assignment, one distinct game per place.

## Scores (exact RULES scorer, log score — 0 is best)

Practice window: 10 archive reads, 2026-11-08 → 2026-11-10. Walk-forward
splits: A = train reads 0–23, window 24–33; B = train 0–19, window 20–33.

| Variant | Practice | A | B | mean(A,B) |
|---|---|---|---|---|
| repeat_last (house baseline) | −43.512 | −56.911 | −53.035 | −54.973 |
| rank_by_playing | −52.601 | −61.701 | −57.847 | −59.774 |
| momentum (defaults) | −45.563 | −46.861 | −55.946 | −51.404 |
| momentum, grid-best (τ12, d2 0.05, s0 2.5, hl 24, iv 3, γ 1.3) | −41.627 | −42.833 | −42.783 | −42.808 |
| fixed 50/50 blend momentum + repeat_last | −38.714 | −46.337 | −44.491 | −45.414 |
| **this entry (time-aware hybrid, bh 12, s0p 0.8, γ 1.3)** | **−39.711** | **−43.328** | **−42.234** | **−42.781** |

Practice per-read: [−31.848, −40.240, −40.725, −38.192, −38.192, −39.445,
−41.582, −41.470, −43.376, −42.038]; top-10 hits 4.5, agreement 36.9.
(The baseline has more top-10 hits, 6.4, but its log score is worse: it
concentrates mass and hits the −9.21 floor whenever the chart moves. Breadth
beats top-10 precision under this scoring rule.)

Grid: 1296 momentum combos; sharpening (γ 1.3) dominates the top 10,
flattening (γ 0.7) the bottom. A second grid over the hybrid (blend horizon ×
persistence sigma × γ) selected the final parameters.

## Why this blend, not the practice-best one

The best practice score in the search (−34.6, heavy persistence, γ 1.0)
collapses on the walk-forward splits (−49.2 mean) — it overfits the practice
window's slow start. The final is 72 hourly reads, where only ~17% are early
hours, so early-read concentration is worth far less than the 10-read
practice metric suggests. The chosen hybrid is the best mean(A,B) in its grid
while also beating pure momentum on practice by +1.9.

## Scenario checks

- **Fast churn**: splits A/B windows include the Nov 5–7 spikes (15–20
  games leaving/joining per read). Momentum holds ≈ −42.8 there; the
  baseline falls to ≈ −55.
- **Brand-new entrants**: of 43 new top-10 entrants in history, 36 came from
  ranks 11–20 and 4 from 31+ — the candidate pool (ranks 1–30 + last 7 days)
  covers them. A game with no history at all cannot be named; it is covered
  by the unnamed share (per-place sums are 0.45–0.71, leaving 0.29–0.55
  spread over ~80 unnamed chart games, ≈ 0.004–0.007 each — far above the
  0.0001 floor).
- **Stalled growth**: velocity decays with τ = 12 h, so a climber that stops
  is treated as flat within about a day.
- **72 h horizon, early vs late**: per-read, the hybrid beats pure momentum
  in the first 3 reads of every split (practice −37.6 vs −38.7, A −37.0 vs
  −37.1, B −32.3 vs −32.9) and holds late (practice −40.6 vs −42.9, B −45.0
  vs −45.5; A −46.1 vs −45.3, the one late-read regression). The baseline
  collapses late (practice read 8: −57.2).

## Verification (practice input)

- Practice log score **−39.711**, identical between the TypeScript entry and
  the Python prototype (max |Δp| = 0.0 over all 163 rows, same top10).
- Determinism: two runs byte-identical (`cmp` clean).
- Runtime ≈ 0.45 s (limit 300 s). Output 8,467 bytes (limit 1,048,576).
- Constraints: 163 probability rows, max 17 rows/place (limit 100),
  per-place sums ≤ 0.71 (limit 1), exactly 10 distinct top10 ids.
- `grade-local` does not grade program/v1 kits (scorer not published in the
  kit); grading used a local scorer implementing the RULES formula exactly,
  validated by reproducing the house baseline's stated practice score
  −43.512039 to 6 decimals.

## Files

- `main.ts` — the entry (TypeScript, stdlib only).
- `LICENSE` — starter licence.
- `README.md` — this file.

Prototypes, backtests and the practice scorer live outside this repo in
`~/workspace/makefun/roblox-forecast/` (`model.py`, `backtest.py`, `tune.py`,
`tune2.py`, `tune3.py`, `tune4.py`, `score.py`, `practice-answer/`).
