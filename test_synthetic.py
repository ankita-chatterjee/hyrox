"""Sanity test with fake keypoints (no video/MediaPipe needed).
Builds a side-on squat set: reps 1,2,4 deep; rep 3 shallow; rep 5 deep but never fully straightens at the end."""
import numpy as np
from run_pipeline import analyse, print_table

fps, W, H = 30, 1000, 1000
def pose(theta_knee_deg, wrist_up):
    th = np.radians(theta_knee_deg)
    A = np.array([0.5, 0.9]); L = 0.2
    K = A + L * np.array([np.sin(np.radians(15)), -np.cos(np.radians(15))])
    d1 = (A - K) / L
    # rotate shin direction by knee angle to get thigh direction (hip behind and above when standing)
    c, s = np.cos(th), np.sin(th)
    d2 = np.array([c * d1[0] + s * d1[1], -s * d1[0] + c * d1[1]])
    Hp = K + L * d2
    trunk_dir = (Hp - K) / L
    S = Hp + 0.25 * trunk_dir  # trunk continues the thigh line, so hip angle ~180
    Wr = np.array([S[0] + 0.1, S[1] - 0.3 * wrist_up + 0.05])
    kp = np.zeros((33, 4)); kp[:, 3] = 0.9
    for idx, p in {23: Hp, 24: Hp, 25: K, 26: K, 27: A, 28: A, 11: S, 12: S + [0.01, 0], 15: Wr, 16: Wr}.items():
        kp[idx, :2] = p
    return kp

THROW = True
def build(depths, top_last=175):
    seq = []
    for i, d in enumerate(depths):
        top = top_last if i == len(depths) - 1 and top_last < 170 else 175
        for t in np.linspace(0, 1, 45):
            ang = top - (top - d) * (0.5 - 0.5 * np.cos(2 * np.pi * t))
            wr = 1.0 if (t > 0.85 and THROW) else 0.0
            seq.append(pose(ang, wr))
    return np.array(seq)

print("=== wall balls: reps 1,2,4 deep; rep 3 shallow; rep 5 never straightens ===")
kp = build([70, 72, 118, 68, 70], top_last=150)
print_table(analyse(dict(kp=kp, fps=fps, width=W, height=H)))

print("\n=== air squats (no throw) ===")
THROW = False
kp = build([70, 72, 70, 68, 70])
print_table(analyse(dict(kp=kp, fps=fps, width=W, height=H)))

print("\n=== air squats, forced ===")
print_table(analyse(dict(kp=kp, fps=fps, width=W, height=H), force=True))
