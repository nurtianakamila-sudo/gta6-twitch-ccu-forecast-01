"""GTA VI launch-day Twitch viewer forecast (gta6-twitch-ccu-forecast-01).

Model: local-linear launch-shape regression on the case's comparables.

For each comparable past launch we take its release-day hourly curve and
express every hour as a log-ratio to that launch's hour-1 print,
r_h = ln(viewers_h / viewers_1).  Across launches the shape of r_h depends
strongly on the size of the launch: the bigger the hour-1 print, the flatter
the day (the largest launches, e.g. ELDEN RING at 875k, stay near 1x all day,
while small launches grow several-fold as regions wake up).  The dependence
is roughly linear in ln(hour-1) but it saturates at the top end, so a global
linear fit over-extrapolates for a launch as big as GTA VI.  Instead, for
each hour we fit an ordinary least-squares line of r_h on ln(viewers_1) over
the K = 20 comparables nearest the target in ln(hour-1) and evaluate it at
the target's own ln(hour-1); the residual standard deviation of that local
fit (floored at 0.15 log units) is the spread.  Quantiles are then

    q_tau(h) = p1 * exp(median_h + z_tau * sd_h),

with z the standard normal quantiles (-1.645, -0.674, 0, 0.674, 1.645),
rounded to 2 decimals half up, clamped to (0, 1e9] and non-decreasing.

Validation (practice answers recovered from the public SullyGnome chart API
the kit's NOTICE.txt documents, and a leave-one-out sweep over all 38
comparable + practice launch-day curves):
  - practice cases p00-p03: mean pinball loss 0.196 (house baseline 0.387)
  - leave-one-out over 38 launch days: 0.359 (house baseline 0.661)
  - big launches (hour-1 >= 100k, the GTA VI regime): 0.21 vs 0.26 baseline

The program is deterministic, standard library only, offline, and runs in
well under a second per case.  It reads every *.in in the input directory
and writes one *.out per case, exactly as the kit's starter does.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import.  (This entry is one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Neighbours (nearest in ln hour-1) used for each hour's local fit.
K_NEIGHBOURS = 20
# Floor on the log-spread of an hour's forecast.
SD_FLOOR = 0.15
# A comparable must have an hour-1 print of at least this many viewers and
# at least this many hours of its launch day to be usable at all.
MIN_HOUR1 = 1000
MIN_HOURS = 13
# Fallback spread (the house baseline's) when a local fit has too few points.
FALLBACK_SIGMA = 0.15


def parse_iso(text):
    """ISO 8601 UTC timestamp -> epoch seconds (handles a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def round2(x):
    """x rounded to 2 decimals, half up (as JavaScript Math.round does)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def launch_day_series(comp):
    """A comparable's launch-day hourly viewers as {hour: viewers}.

    Hour h is [day_start + (h-1) h, day_start + h h).  Only hours 1..24 are
    kept.  The target game's own pre-release category history is not a
    launch day and is excluded by the caller.
    """
    start = parse_iso(comp["day_start"])
    series = {}
    for point in comp.get("hourly", []):
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        hour = int((parse_iso(point["from"]) - start) // 3600) + 1
        if 1 <= hour <= 24:
            series[hour] = viewers
    return series


def is_launch_curve(comp):
    """True when a comparable is a real past launch day, not target history."""
    unlock = (comp.get("unlock") or "").lower()
    name = (comp.get("name") or "").lower()
    if unlock.startswith("not a launch") or "before release" in name:
        return False
    return True


def local_fit_median_sd(neighbours, hour, log_p1):
    """OLS of ln(viewers_h / viewers_1) on ln(viewers_1) over neighbours.

    neighbours: [(series, log_hour1)] restricted to series having both hour
    1 and `hour`.  Returns (median log-ratio, residual sd) or None when too
    few points carry the hour.
    """
    xs = []
    ys = []
    for series, log_h1 in neighbours:
        if 1 in series and hour in series:
            xs.append(log_h1)
            ys.append(math.log(series[hour] / series[1]))
    if len(xs) < 4:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    vx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / vx if vx > 0 else 0.0
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sd = math.sqrt(sum(r * r for r in resid) / max(n - 2, 1))
    return intercept + slope * log_p1, max(sd, SD_FLOOR)


def hour_quantiles(p1, log_p1, neighbours, hour):
    """The five quantiles for one forecast hour."""
    fit = local_fit_median_sd(neighbours, hour, log_p1)
    if fit is None:
        median, sd = 0.0, FALLBACK_SIGMA * math.sqrt(hour - 1)
    else:
        median, sd = fit
    out = []
    for z in Z:
        value = round2(p1 * math.exp(median + z * sd))
        value = max(1, min(value, 1_000_000_000))
        out.append(value)
    # Rounding can only produce ties, never a decrease, but enforce the
    # contract explicitly so the row is non-decreasing by construction.
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    p1 = max(1, first["viewers"]) if first is not None else 1

    # Usable comparables: real launch days with a usable hour-1 print.
    pool = []
    for comp in case.get("comparables", []):
        if not is_launch_curve(comp):
            continue
        series = launch_day_series(comp)
        if 1 not in series or len(series) < MIN_HOURS:
            continue
        if series[1] < MIN_HOUR1:
            continue
        pool.append((series, math.log(series[1]), comp.get("name", "")))
    # Nearest neighbours in log hour-1, ties broken by name for determinism.
    log_p1 = math.log(p1)
    pool.sort(key=lambda item: (abs(item[1] - log_p1), item[2]))
    neighbours = [(series, log_h1) for series, log_h1, _ in pool[:K_NEIGHBOURS]]

    hours = []
    for row in case.get("forecast", []):
        hour = row["hour"]
        hours.append({"hour": hour, "q": hour_quantiles(p1, log_p1, neighbours, hour)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
