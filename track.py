"""Step 3 - Track: run MediaPipe Pose on every frame and cache keypoints to disk.

Saved array `kp` has shape (n_frames, 33, 4): x, y (normalised 0-1), z, visibility.
Frames where no person is detected are NaN with visibility 0.
Everything downstream reads this file, so you can retune rules without re-tracking.
"""
import numpy as np
from config import TRACKER


def track_video(video_path: str, out_npz: str, show_progress: bool = True) -> dict:
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    with mp.solutions.pose.Pose(static_image_mode=False, smooth_landmarks=True, **TRACKER) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                frames.append([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks.landmark])
            else:
                frames.append([[np.nan, np.nan, np.nan, 0.0]] * 33)
            i += 1
            if show_progress and i % 50 == 0:
                print(f"  tracked {i}/{n_total} frames", end="\r")
    cap.release()

    kp = np.asarray(frames, dtype=np.float32)
    np.savez_compressed(out_npz, kp=kp, fps=fps, width=w, height=h)
    detected = float(np.mean(~np.isnan(kp[:, 0, 0])))
    print(f"\nTracked {len(kp)} frames @ {fps:.1f} fps, person detected in {detected:.0%} of frames -> {out_npz}")
    return dict(kp=kp, fps=fps, width=w, height=h)


def load_keypoints(npz_path: str) -> dict:
    d = np.load(npz_path)
    return dict(kp=d["kp"], fps=float(d["fps"]), width=int(d["width"]), height=int(d["height"]))
