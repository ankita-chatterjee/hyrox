"""Step 5 - Wall-ball rules. Every threshold comes from config.RULES.

Rule status: pass | fail | not_assessable   (a rule function may return None = "does not apply to this rep")
Rep verdict:
  not_proper      - at least one NON-advisory rule failed (failed rules listed)
  not_assessable  - nothing failed, but a CORE rule (squat depth) could not be judged
  proper          - no failures and all core rules judged and passed
Advisory rules (coaching) are reported in `advisories` and never change the verdict.

Each rep is split into phases:  start(top) -> DESCENT -> bottom -> DRIVE/THROW -> end(top)
"""
from dataclasses import dataclass, asdict
import numpy as np
from config import RULES, VIEW
from segment import Signals, Rep


@dataclass
class RuleResult:
    rule: str
    status: str
    value: float = None
    detail: str = ""


def view_check(kp, w, h, sig: Signals) -> dict:
    sh_sep = np.abs(kp[:, 11, 0] - kp[:, 12, 0]) * w
    ok = (kp[:, 11, 3] > 0.5) & (kp[:, 12, 3] > 0.5) & ~np.isnan(sh_sep)
    torso = np.linalg.norm(sig.pts["shoulder"] - sig.pts["hip"], axis=1)
    if ok.sum() < 5:
        return dict(side_on=None, ratio=None)
    ratio = float(np.median(sh_sep[ok] / (torso[ok] + 1e-9)))
    return dict(side_on=ratio <= VIEW["side_on_max_ratio"], ratio=ratio)


def make_context(sig: Signals) -> dict:
    return dict(
        thigh_len=float(np.median(np.linalg.norm(sig.pts["hip"] - sig.pts["knee"], axis=1))),
        torso_len=float(np.median(np.linalg.norm(sig.pts["shoulder"] - sig.pts["hip"], axis=1))),
    )


def _win(frame, fps, n, half_s=0.1):
    h = max(1, int(half_s * fps))
    return slice(max(0, frame - h), min(n, frame + h + 1))


def _vis_ok(sig, joints, sl, min_vis):
    return all(np.mean(sig.vis[j][sl]) >= min_vis for j in joints)


def _na(rule, why):
    return RuleResult(rule, "not_assessable", None, why)


def _side_na(rule, view):
    if RULES[rule].get("needs_side_view") and view["side_on"] is not True:
        return _na(rule, f"view not side-on (shoulder/torso ratio={view['ratio']}); not judged")
    return None


def wrist_reach(sig, rep, ctx, min_vis=0.5):
    """Max wrist height above shoulder (in torso lengths) in the drive phase bottom->end. None if wrist unreliable."""
    if sig.pts.get("wrist") is None:
        return None
    sl = slice(rep.bottom, rep.end + 1)
    if np.mean(sig.vis["wrist"][sl]) < min_vis:
        return None
    return float(np.max((sig.pts["shoulder"][sl, 1] - sig.pts["wrist"][sl, 1]) / ctx["torso_len"]))


# ---------------- individual rules ----------------
def rule_start_standing(sig, rep, view, ctx):
    if rep.index != 0:
        return None
    c = RULES["start_standing"]
    sl = _win(rep.start, sig.fps, len(sig.knee_angle), 0.15)
    if not _vis_ok(sig, ("hip", "knee", "ankle", "shoulder"), sl, c["min_visibility"]):
        return _na("start_standing", "leg/torso low confidence at the start")
    k, hp = float(np.max(sig.knee_angle[sl])), float(np.max(sig.hip_angle[sl]))
    ok = k >= c["min_knee_angle"] and hp >= c["min_hip_angle"]
    return RuleResult("start_standing", "pass" if ok else "fail", min(k, hp),
                      f"standing start: knee {k:.0f} / hip {hp:.0f} deg (need >= {c['min_knee_angle']}/{c['min_hip_angle']})")


def rule_squat_depth(sig, rep, view, ctx):
    c = RULES["squat_depth"]
    na = _side_na("squat_depth", view)
    if na:
        return na
    sl = _win(rep.bottom, sig.fps, len(sig.knee_angle))
    if not _vis_ok(sig, ("hip", "knee"), sl, c["min_visibility"]):
        return _na("squat_depth", "hip/knee low confidence at the bottom of the rep")
    margin = float(np.max((sig.pts["hip"][sl, 1] - sig.pts["knee"][sl, 1]) / ctx["thigh_len"]))
    limit = -c["tolerance_thigh"]
    border = abs(margin - limit) <= c["borderline_band"]
    return RuleResult("squat_depth", "pass" if margin >= limit else "fail", margin,
                      f"hip-vs-knee depth {margin:+.2f} thigh-lengths (need >= {limit:+.2f})" + (" [borderline]" if border else ""))


def rule_full_extension(sig, rep, view, ctx):
    c = RULES["full_extension"]
    na = _side_na("full_extension", view)
    if na:
        return na
    sl = _win(rep.end, sig.fps, len(sig.knee_angle), 0.15)
    if not _vis_ok(sig, ("hip", "knee", "ankle", "shoulder"), sl, c["min_visibility"]):
        return _na("full_extension", "leg/torso low confidence at the top of the rep")
    k, hp = float(np.max(sig.knee_angle[sl])), float(np.max(sig.hip_angle[sl]))
    ok = k >= c["min_knee_angle"] and hp >= c["min_hip_angle"]
    return RuleResult("full_extension", "pass" if ok else "fail", min(k, hp),
                      f"top of rep: knee {k:.0f} / hip {hp:.0f} deg (need >= {c['min_knee_angle']}/{c['min_hip_angle']})")


def rule_throw_reach(sig, rep, view, ctx):
    c = RULES["throw_reach"]
    reach = wrist_reach(sig, rep, ctx, c["min_visibility"])
    if reach is None:
        return _na("throw_reach", "wrist not tracked reliably (likely hidden by the ball)")
    ok = reach >= c["wrist_above_shoulder_torso"]
    return RuleResult("throw_reach", "pass" if ok else "fail", reach,
                      f"wrist peaked {reach:+.2f} torso-lengths above shoulder (need >= {c['wrist_above_shoulder_torso']})")


def rule_arm_only_throw(sig, rep, view, ctx):
    c = RULES["arm_only_throw"]
    na = _side_na("arm_only_throw", view)
    if na:
        return na
    if sig.pts.get("wrist") is None:
        return _na("arm_only_throw", "wrist not tracked")
    sl = slice(rep.bottom, rep.end + 1)
    if np.mean(sig.vis["wrist"][sl]) < c["min_visibility"]:
        return _na("arm_only_throw", "wrist low confidence")
    above = (sig.pts["shoulder"][sl, 1] - sig.pts["wrist"][sl, 1]) > 0
    if not above.any():
        return None   # no throw seen; throw_reach already reports that
    f = rep.bottom + int(np.argmax(above))
    k = float(sig.knee_angle[f])
    ok = k >= c["min_knee_angle_when_arms_rise"]
    return RuleResult("arm_only_throw", "pass" if ok else "fail", k,
                      f"knee {k:.0f} deg when wrists passed shoulder (need >= {c['min_knee_angle_when_arms_rise']}): "
                      + ("legs drove first" if ok else "arms started before legs"))


def rule_ball_at_chest_bottom(sig, rep, view, ctx):
    c = RULES["ball_at_chest_bottom"]
    if sig.pts.get("wrist") is None:
        return _na("ball_at_chest_bottom", "wrist not tracked")
    sl = _win(rep.bottom, sig.fps, len(sig.knee_angle))
    if np.mean(sig.vis["wrist"][sl]) < c["min_visibility"]:
        return _na("ball_at_chest_bottom", "wrist low confidence")
    s, hp, wr = sig.pts["shoulder"][sl, 1], sig.pts["hip"][sl, 1], sig.pts["wrist"][sl, 1]
    frac = float(np.median((wr - s) / (hp - s + 1e-9)))
    ok = c["min_frac"] <= frac <= c["max_frac"]
    return RuleResult("ball_at_chest_bottom", "pass" if ok else "fail", frac,
                      f"hands at {frac:.2f} of shoulder->hip height at bottom (need {c['min_frac']}-{c['max_frac']})")


def rule_target_hit(sig, rep, view, ctx):
    return _na("target_hit", RULES["target_hit"]["reason"])


def rule_catch_after_bounce(sig, rep, view, ctx):
    return _na("catch_after_bounce", RULES["catch_after_bounce"]["reason"])


RULE_FUNCS = {
    "start_standing": rule_start_standing,
    "squat_depth": rule_squat_depth,
    "full_extension": rule_full_extension,
    "throw_reach": rule_throw_reach,
    "arm_only_throw": rule_arm_only_throw,
    "ball_at_chest_bottom": rule_ball_at_chest_bottom,
    "target_hit": rule_target_hit,
    "catch_after_bounce": rule_catch_after_bounce,
}


def judge_rep(sig: Signals, rep: Rep, view: dict, ctx: dict) -> dict:
    results = []
    for name, cfg in RULES.items():
        if cfg.get("enabled"):
            r = RULE_FUNCS[name](sig, rep, view, ctx)
            if r is not None:
                results.append(r)
    adv = {n for n, c in RULES.items() if c.get("advisory")}
    failed = [r.rule for r in results if r.status == "fail" and r.rule not in adv]
    advisories = [r.rule for r in results if r.status == "fail" and r.rule in adv]
    core_na = [r.rule for r in results if r.status == "not_assessable" and RULES[r.rule].get("core")]
    verdict = "not_proper" if failed else ("not_assessable" if core_na else "proper")
    fps = sig.fps
    return dict(
        rep=rep.index + 1,
        start_frame=rep.start, bottom_frame=rep.bottom, end_frame=rep.end,
        # phase timing, used later for tempo / fatigue analysis
        duration_s=round((rep.end - rep.start) / fps, 2),
        descent_s=round((rep.bottom - rep.start) / fps, 2),
        drive_s=round((rep.end - rep.bottom) / fps, 2),
        min_knee_angle=round(float(sig.knee_angle[rep.bottom]), 1),
        verdict=verdict,
        failed_rules=failed,
        advisories=advisories,
        not_assessable=[r.rule for r in results if r.status == "not_assessable"],
        rules=[asdict(r) for r in results],
    )
