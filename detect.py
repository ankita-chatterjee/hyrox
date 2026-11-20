"""Exercise detection: is this clip actually wall balls?  (Rule-based, no model.)

Wall-ball signature:
  1. >= min_reps squat cycles                      (rules out walking, running, single movements)
  2. cycle length steady and in a plausible range  (rules out random motion)
  3. hands go overhead in most cycles              (separates wall balls from plain air squats)
  4. both legs move together                       (separates from alternating-leg lunges)
Labels: wall_balls | squats_no_throw | not_wall_balls | uncertain | unknown
"""
import numpy as np
from config import DETECTION as D
from segment import leg_asymmetry
from rules import wrist_reach


def detect_exercise(data, sig, reps, ctx) -> dict:
    fps = sig.fps
    checks, notes = {}, []

    checks["enough_cycles"] = len(reps) >= D["min_reps"]
    notes.append(f"{len(reps)} squat cycles found (need >= {D['min_reps']})")

    if len(reps) >= 3:
        periods = np.diff([r.bottom for r in reps]) / fps
        m, cv = float(np.mean(periods)), float(np.std(periods) / (np.mean(periods) + 1e-9))
        checks["steady_tempo"] = D["min_period_s"] <= m <= D["max_period_s"] and cv <= D["max_period_cv"]
        notes.append(f"cycle length {m:.1f}s, variation {cv:.2f}")
    else:
        checks["steady_tempo"] = None

    reaches = [wrist_reach(sig, r, ctx) for r in reps]
    valid = [x for x in reaches if x is not None]
    if valid:
        frac = float(np.mean([x >= D["overhead_reach_torso"] for x in valid]))
        checks["hands_overhead"] = frac >= D["min_overhead_frac"]
        notes.append(f"hands overhead in {frac:.0%} of cycles")
    else:
        checks["hands_overhead"] = None
        notes.append("wrist not reliably tracked, overhead check skipped")

    asym, used = leg_asymmetry(data)
    if asym is not None and used >= D["min_leg_vis_frac"]:
        checks["legs_in_sync"] = asym <= D["max_leg_asym_deg"]
        notes.append(f"left/right knee angle differ by {asym:.0f} deg (median)")
    else:
        checks["legs_in_sync"] = None
        notes.append("far leg not visible enough, leg-sync check skipped")

    evaluated = [v for v in checks.values() if v is not None]
    confidence = float(np.mean(evaluated)) if evaluated else 0.0

    if not checks["enough_cycles"]:
        label = "unknown"
    elif checks["hands_overhead"] is False:
        label = "squats_no_throw"
    elif checks["legs_in_sync"] is False:
        label = "not_wall_balls"
    elif checks["steady_tempo"] is False:
        label = "uncertain"
    elif checks["hands_overhead"] is None:
        label = "uncertain"
    else:
        label = "wall_balls"
    return dict(label=label, confidence=round(confidence, 2), checks=checks, notes=notes)
