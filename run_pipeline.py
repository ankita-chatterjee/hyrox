"""Usage:
    python run_pipeline.py clips/wallballs_01.mp4 --out outputs/
    python run_pipeline.py clips/wallballs_01.mp4 --out outputs/ --retrack   # force re-run of MediaPipe
Re-running after editing config.py reuses the cached keypoints, so tuning is fast.
"""
import argparse, json, os
from track import track_video, load_keypoints
from segment import build_signals, segment_reps
from rules import view_check, make_context, judge_rep
from detect import detect_exercise


def analyse(data: dict, force: bool = False) -> dict:
    sig = build_signals(data)
    if not sig.usable:
        return dict(ok=False, message=sig.note, reps=[])
    view = view_check(data["kp"], data["width"], data["height"], sig)
    reps = segment_reps(sig)
    ctx = make_context(sig)
    detection = detect_exercise(data, sig, reps, ctx)
    warnings = []
    if view["side_on"] is not True:
        warnings.append(f"Camera does not look side-on (ratio={view['ratio']}); depth/extension not assessable.")
    if not reps:
        warnings.append("No reps detected. Check the camera angle, clip trim, or SEGMENTATION thresholds.")
    judged = detection["label"] in ("wall_balls", "uncertain") or force
    if not judged:
        warnings.append(f"Clip does not look like wall balls (label={detection['label']}); verdicts skipped. Use --force to override.")
    verdicts = [judge_rep(sig, r, view, ctx) for r in reps] if judged else []
    bounds = [dict(rep=r.index + 1, start_frame=r.start, bottom_frame=r.bottom, end_frame=r.end) for r in reps]
    return dict(ok=True, fps=data["fps"], side_used=sig.side, view=view, detection=detection,
                warnings=warnings, reps=verdicts, reps_detected=len(reps), rep_bounds=bounds, _sig=sig)


def print_table(res: dict):
    if not res["ok"]:
        print("FAILED:", res["message"]); return
    for w in res["warnings"]:
        print("WARNING:", w)
    d = res["detection"]
    print(f"\nExercise detection: {d['label']} (confidence {d['confidence']})")
    for n in d["notes"]:
        print("  -", n)
    print(f"Leg used: {res['side_used']}   cycles detected: {res['reps_detected']}\n")
    print(f"{'#':>3} {'time(s)':>8} {'verdict':<15} failed / not assessable")
    for r in res["reps"]:
        t = r["bottom_frame"] / res["fps"]
        extra = ", ".join(r["failed_rules"]) or "-"
        adv = ", ".join(r["advisories"])
        na = ", ".join(r["not_assessable"])
        print(f"{r['rep']:>3} {t:>8.1f} {r['verdict']:<15} {extra}"
              + (f"   [tip: {adv}]" if adv else "") + (f"   [n/a: {na}]" if na else ""))
    n = len(res["reps"])
    proper = sum(r["verdict"] == "proper" for r in res["reps"])
    bad = sum(r["verdict"] == "not_proper" for r in res["reps"])
    print(f"\nproper {proper} | not proper {bad} | not assessable {n - proper - bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--retrack", action="store_true")
    ap.add_argument("--force", action="store_true", help="judge reps even if the clip is not detected as wall balls")
    ap.add_argument("--no-video", action="store_true", help="skip rendering the annotated video")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.video))[0]
    p = lambda suffix: os.path.join(a.out, f"{stem}_{suffix}")
    npz = p("keypoints.npz")
    data = track_video(a.video, npz) if (a.retrack or not os.path.exists(npz)) else load_keypoints(npz)
    res = analyse(data, force=a.force)
    print_table(res)
    if res["ok"]:
        from render import frame_labels, write_frames_csv, write_reps_csv, render_video
        sig = res.pop("_sig")
        write_frames_csv(p("frames.csv"), frame_labels(res, len(data["kp"]), sig))
        if res["reps"]:
            write_reps_csv(p("reps.csv"), res)
        if not a.no_video:
            print("Rendering annotated video ...")
            render_video(a.video, data, res, p("annotated.mp4"), sig)
    with open(p("verdicts.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {a.out}/: {stem}_annotated.mp4, {stem}_frames.csv, {stem}_reps.csv, {stem}_verdicts.json, {stem}_keypoints.npz")
