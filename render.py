"""Outputs: annotated video (skeleton + live rep counter + per-rep verdict) and data labels (CSV/JSON).

Files written per clip (stem = video file name):
  <stem>_annotated.mp4   video with overlay
  <stem>_frames.csv      one row per frame: rep_count, current_rep, phase, knee/hip angle, last verdict
  <stem>_reps.csv        one row per rep: timings, verdict, failed rules, each rule's status and measured value
  <stem>_verdicts.json   everything (detection, view, warnings, per-rep detail)
"""
import csv, os, shutil, subprocess
import cv2
import numpy as np
from config import RULES

CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
               (23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (29, 31), (28, 30), (30, 32), (27, 31), (28, 32)]
TRACKED = {"L": {11, 23, 25, 27, 13, 15}, "R": {12, 24, 26, 28, 14, 16}}
COLOR = dict(proper=(70, 200, 70), not_proper=(60, 60, 235), not_assessable=(0, 165, 255), none=(210, 210, 210))
LABEL = dict(proper="PROPER", not_proper="NO REP", not_assessable="NOT ASSESSABLE")
SHOW_VERDICT_S = 2.0   # how long a verdict stays on screen after the rep completes


# ---------------------------------------------------------------- labels
def frame_labels(res: dict, n_frames: int, sig=None) -> list:
    fps = res["fps"]
    verdict_by_rep = {r["rep"]: r for r in res["reps"]}
    bounds = res["rep_bounds"]
    out = []
    for f in range(n_frames):
        done = [b for b in bounds if b["end_frame"] <= f]
        cur = next((b for b in bounds if b["start_frame"] <= f < b["end_frame"]), None)
        phase = "idle" if cur is None else ("descent" if f < cur["bottom_frame"] else "drive")
        last = verdict_by_rep.get(done[-1]["rep"]) if done else None
        judged = [verdict_by_rep[b["rep"]]["verdict"] for b in done if b["rep"] in verdict_by_rep]
        out.append(dict(
            frame=f, time_s=round(f / fps, 3),
            rep_count=len(done), current_rep=cur["rep"] if cur else "", phase=phase,
            knee_angle=round(float(sig.knee_angle[f]), 1) if sig is not None and f < len(sig.knee_angle) else "",
            hip_angle=round(float(sig.hip_angle[f]), 1) if sig is not None and f < len(sig.hip_angle) else "",
            last_rep_verdict=last["verdict"] if last else "",
            last_rep_failed=";".join(last["failed_rules"]) if last else "",
            proper_so_far=judged.count("proper"), not_proper_so_far=judged.count("not_proper"),
        ))
    return out


def write_frames_csv(path, labels):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(labels[0].keys()))
        w.writeheader(); w.writerows(labels)


def write_reps_csv(path, res):
    rule_names = [n for n, c in RULES.items() if c.get("enabled")]
    cols = ["rep", "start_s", "bottom_s", "end_s", "duration_s", "descent_s", "drive_s", "min_knee_angle",
            "verdict", "failed_rules", "advisories", "not_assessable"]
    for n in rule_names:
        cols += [f"{n}_status", f"{n}_value"]
    fps = res["fps"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for r in res["reps"]:
            row = dict(rep=r["rep"], start_s=round(r["start_frame"] / fps, 2), bottom_s=round(r["bottom_frame"] / fps, 2),
                       end_s=round(r["end_frame"] / fps, 2), duration_s=r["duration_s"], descent_s=r["descent_s"],
                       drive_s=r["drive_s"], min_knee_angle=r["min_knee_angle"], verdict=r["verdict"],
                       failed_rules=";".join(r["failed_rules"]), advisories=";".join(r["advisories"]),
                       not_assessable=";".join(r["not_assessable"]))
            for rr in r["rules"]:
                row[f"{rr['rule']}_status"] = rr["status"]
                row[f"{rr['rule']}_value"] = "" if rr["value"] is None else round(rr["value"], 3)
            w.writerow(row)


# ---------------------------------------------------------------- drawing
def _text(img, s, org, scale, color, thick=2):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return th + 16


def _skeleton(img, kp_f, side, w, h):
    pts = {}
    for j in range(33):
        x, y, _, v = kp_f[j]
        if not np.isnan(x) and v >= 0.3:
            pts[j] = (int(x * w), int(y * h))
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            hot = a in TRACKED[side] and b in TRACKED[side]
            cv2.line(img, pts[a], pts[b], (0, 220, 255) if hot else (255, 255, 255), 4 if hot else 2, cv2.LINE_AA)
    for j, p in pts.items():
        cv2.circle(img, p, 5, (0, 220, 255) if j in TRACKED[side] else (200, 200, 200), -1, cv2.LINE_AA)
    return pts


def render_video(video_path, data, res, out_path, sig=None):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or res["fps"]
    tmp = out_path + ".tmp.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = len(data["kp"])
    labels = frame_labels(res, n, sig)
    verdict_by_rep = {r["rep"]: r for r in res["reps"]}
    s = max(0.6, h / 900)                       # text scale relative to video size
    det = res["detection"]
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if f < n:
            L = labels[f]
            pts = _skeleton(img, data["kp"][f], res["side_used"], w, h)
            # knee angle next to the tracked knee
            kj = 25 if res["side_used"] == "L" else 26
            if kj in pts and L["knee_angle"] != "":
                cv2.putText(img, f"{L['knee_angle']:.0f}", (pts[kj][0] + 12, pts[kj][1]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7 * s, (0, 220, 255), 2, cv2.LINE_AA)
            y = int(36 * s)
            y += _text(img, f"REPS {L['rep_count']}", (14, y), 1.2 * s, (255, 255, 255), 3)
            y += _text(img, f"proper {L['proper_so_far']}   no rep {L['not_proper_so_far']}", (14, y + 4), 0.7 * s, (255, 255, 255), 2)
            y += _text(img, f"phase: {L['phase']}   exercise: {det['label']}", (14, y + 4), 0.6 * s, (210, 210, 210), 1)
            # verdict banner for the most recently completed rep
            last = next((b for b in reversed(res["rep_bounds"]) if b["end_frame"] <= f), None)
            if last and (f - last["end_frame"]) <= SHOW_VERDICT_S * fps:
                v = verdict_by_rep.get(last["rep"])
                if v:
                    c = COLOR[v["verdict"]]
                    y += _text(img, f"REP {v['rep']}: {LABEL[v['verdict']]}", (14, y + 10), 1.0 * s, c, 3)
                    if v["failed_rules"]:
                        y += _text(img, "failed: " + ", ".join(v["failed_rules"]), (14, y + 8), 0.65 * s, c, 2)
                    elif v["verdict"] == "not_assessable":
                        y += _text(img, "depth not judgeable from this view", (14, y + 8), 0.65 * s, c, 2)
                    if v["advisories"]:
                        _text(img, "tip: " + ", ".join(v["advisories"]), (14, y + 8), 0.55 * s, (0, 220, 255), 1)
            for i, wmsg in enumerate(res["warnings"][:2]):
                _text(img, "! " + wmsg[:95], (14, h - 14 - int(i * 34 * s)), 0.5 * s, (0, 165, 255), 1)
        vw.write(img)
        f += 1
    cap.release(); vw.release()
    # re-encode to H.264 so it plays in browsers/phones, if ffmpeg exists
    if shutil.which("ffmpeg"):
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path])
        if r.returncode == 0:
            os.remove(tmp); return out_path
    os.replace(tmp, out_path)
    return out_path
