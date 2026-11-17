# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nurtianakamila-sudo/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/nurtianakamila-sudo/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for Grand Theft Auto VI's launch-day Twitch audience: given the first hour's viewer count on 2026-11-19 (UTC), it predicts the other 23 hours as five quantiles each. Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty ($250 in USDC). Entry by @kamila-nurtiana-terserah-sera.

## How it works

`main.py` (Python 3.13, standard library only, no network, deterministic) reads the case file `/input/<id>.in` — the hour-1 print plus the hourly launch-day curves of 38–42 past big game launches — and writes `/output/<id>.out` with 23 rows, one per hour 2–24.

The model is a **local-linear launch-shape regression**: for each comparable launch, every hour is expressed as a log-ratio to that launch's hour-1 print. Launch-day shapes depend strongly on launch size — the biggest launches stay near their hour-1 level all day, while smaller ones grow several-fold as regions wake up — and the dependence saturates at the top end, so a global linear fit would over-extrapolate for a launch as big as GTA VI. Instead, for each hour the program fits a least-squares line of the log-ratio on log hour-1 over the 20 comparables nearest the target in log hour-1, evaluates it at the target's own hour-1, and uses the local residual spread (floored at 0.15 log units) for the quantiles:

```
q_tau(h) = p1 * exp(median_h + z_tau * sd_h)
```

with z = (-1.645, -0.674, 0, 0.674, 1.645) for the 5%, 25%, 50%, 75% and 95% levels.

## Validation

Practice answers were recovered from the public SullyGnome chart API the kit's `input/NOTICE.txt` documents, and cross-checked against the house baseline's published practice loss (0.387, reproduced exactly at 0.3871).

| Method | Practice loss (p00–p03) | Leave-one-out, 38 launch days |
|---|---|---|
| House baseline (persistence) | 0.387 | 0.661 |
| This entry | **0.196** | **0.359** |

Per practice case: p00 0.223, p01 0.226, p02 0.138, p03 0.199 (typical miss 26–51%, 90% coverage 0.91–1.00). On big launches only (hour-1 ≥ 100k, the GTA VI regime) the leave-one-out loss is 0.21 vs 0.26 for the baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Two runs on the same case write byte-identical output; a run of all four practice cases takes about 0.2 s.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
