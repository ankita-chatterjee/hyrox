"""Step 4 - Segment reps from the keypoint signal.

Signal = knee angle (hip-knee-ankle) of the better-visible leg.
Each rep is one valley (the bottom of a squat). Rep boundaries are the angle peaks (the standing/throw
positions) on either side of the valley. Using peaks as boundaries means a rep is still counted if the
athlete never fully straightens between reps (fatigue), and that incomplete extension is then judged by
the full_extension rule instead of silently merging two reps.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from config import SIGNAL, SEGMENTATION

# MediaPipe Pose landmark indices
IDX = {
    "L": dict(shoulder=11, elbow=13, wrist=15, hip=23, knee=25, ankle=27),
    "R": dict(shoulder=12, elbow=14, wrist=16, hip=24, knee=26, ankle=28),
}


@dataclass
class Rep:
    index: int
    start: int    # frame: top of the previous rep / start of set
    bottom: int   # frame: deepest point of squat
    end: int      # frame: top of this rep (shared with next rep's start)


@dataclass
class Signals:
    side: str
    fps: float
    pts: dict        # joint name -> (N,2) pixel coords, gaps interpolated
    vis: dict        # joint name -> (N,) raw visibility
    knee_angle: np.ndarray
    hip_angle: np.ndarray
    usable: bool
    note: str = ""


def angle_deg(a, b, c):
    """Interior angle at b for points a-b-c, each (N,2). Returns (N,) degrees."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def choose_side(kp) -> str:
    """Use the leg the camera sees best (side-on videos: the near leg)."""
    score = {}
    for s, ix in IDX.items():
        score[s] = float(np.nanmean([kp[:, ix[j], 3] for j in ("shoulder", "hip", "knee", "ankle")]))
    return max(score, key=score.get)


def _clean(series, vis, min_vis):
    """Mask low-confidence/NaN samples and linearly interpolate across them."""
    bad = np.isnan(series) | (vis < min_vis)
    if bad.mean() > SIGNAL["max_missing_frac"] or (~bad).sum() < 3:
        return None
    x = np.arange(len(series))
    return np.interp(x, x[~bad], series[~bad])


def _smooth(sig, fps):
    win = max(5, int(SIGNAL["smooth_window_s"] * fps) | 1)  # odd
    win = min(win, len(sig) - (1 - len(sig) % 2))
    return savgol_filter(sig, win, SIGNAL["smooth_polyorder"]) if win > SIGNAL["smooth_polyorder"] else sig


def build_signals(data: dict, side: str = None) -> Signals:
    kp, fps, w, h = data["kp"], data["fps"], data["width"], data["height"]
    side = side or choose_side(kp)
    ix = IDX[side]
    pts, vis = {}, {}
    for name, j in ix.items():
        vis[name] = np.nan_to_num(kp[:, j, 3])
        xs = _clean(kp[:, j, 0] * w, vis[name], SIGNAL["min_visibility"])
        ys = _clean(kp[:, j, 1] * h, vis[name], SIGNAL["min_visibility"])
        if xs is None or ys is None:
            pts[name] = None
        else:
            pts[name] = np.stack([xs, ys], axis=1)
    need = ("shoulder", "hip", "knee", "ankle")
    if any(pts[n] is None for n in need):
        return Signals(side, fps, pts, vis, None, None, False,
                       "Too many low-confidence frames on the chosen leg; cannot segment reps.")
    knee = _smooth(angle_deg(pts["hip"], pts["knee"], pts["ankle"]), fps)
    hip = _smooth(angle_deg(pts["shoulder"], pts["hip"], pts["knee"]), fps)
    return Signals(side, fps, pts, vis, knee, hip, True)


def segment_reps(sig: Signals) -> list:
    cfg = SEGMENTATION
    a = sig.knee_angle
    valleys, _ = find_peaks(-a, prominence=cfg["min_prominence"], distance=max(1, int(cfg["min_rep_s"] * sig.fps)))
    valleys = [int(v) for v in valleys if a[v] <= cfg["valley_max_angle"]]
    if not valleys:
        return []
    # boundaries: the highest-angle frame (most upright) between consecutive valleys
    bounds = [int(np.argmax(a[: valleys[0] + 1]))]
    for v0, v1 in zip(valleys[:-1], valleys[1:]):
        bounds.append(v0 + int(np.argmax(a[v0: v1 + 1])))
    bounds.append(valleys[-1] + int(np.argmax(a[valleys[-1]:])))
    return [Rep(i, bounds[i], v, bounds[i + 1]) for i, v in enumerate(valleys)]


def leg_asymmetry(data: dict):
    """Median |left knee angle - right knee angle| over frames where both legs are visible.
    Returns (value_deg or None, fraction_of_frames_used). Large value => legs doing different things (lunges)."""
    kp, w, h = data["kp"], data["width"], data["height"]
    def ang(side):
        ix = IDX[side]
        p = lambda j: np.stack([kp[:, ix[j], 0] * w, kp[:, ix[j], 1] * h], axis=1)
        v = np.minimum.reduce([kp[:, ix[j], 3] for j in ("hip", "knee", "ankle")])
        return angle_deg(p("hip"), p("knee"), p("ankle")), v
    a_l, v_l = ang("L"); a_r, v_r = ang("R")
    ok = (v_l >= SIGNAL["min_visibility"]) & (v_r >= SIGNAL["min_visibility"]) & ~np.isnan(a_l) & ~np.isnan(a_r)
    frac = float(ok.mean())
    if ok.sum() < 5:
        return None, frac
    return float(np.median(np.abs(a_l[ok] - a_r[ok]))), frac
