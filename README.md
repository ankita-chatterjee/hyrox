# HYROX Wall Balls Form-Check (rule-based)

A rule-based pipeline that watches a video of a HYROX **Wall Balls** set, counts the repetitions, judges each rep as **proper / not proper / not assessable**, and names the rule that failed. It also checks whether the clip is wall balls at all, and writes an annotated video plus CSV/JSON labels.

No model training or fine-tuning is used anywhere. Every verdict comes from thresholds on body-tracking keypoints, and every threshold lives in one file (`config.py`).



---



---

## Quick start

**Requirements:** Python 3.9 to 3.12 (MediaPipe wheels are unreliable on newer versions). `ffmpeg` is optional but recommended, because it re-encodes the output video to H.264 so it plays in browsers and on phones.

```bash
# 1. Go to the project folder
cd hyrox_formcheck

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Sanity check without a video (uses fake keypoints; MediaPipe not needed)
python test_synthetic.py

# 5. Run on a real clip
python run_pipeline.py clips/wallballs_01.mp4 --out outputs/
```

### Command-line options

| Option | Effect |
|---|---|
| `--out DIR` | Output folder (default `outputs`). |
| `--retrack` | Re-run MediaPipe even if cached keypoints exist. |
| `--force` | Judge reps even if the clip is not detected as wall balls. |
| `--no-video` | Skip rendering the annotated video (faster). |

The first run on a clip runs MediaPipe and caches the keypoints. After you edit `config.py`, rerun the same command: it reuses the cache, so tuning takes seconds.

---

## Outputs

For a clip called `wallballs_01.mp4`, the output folder contains:

| File | Contents |
|---|---|
| `wallballs_01_annotated.mp4` | Video with skeleton overlay (tracked leg in yellow), live knee angle, live rep counter, proper/no-rep totals, and a verdict banner for 2 s after each rep (green `PROPER`, red `NO REP` with failed rules, orange `NOT ASSESSABLE`). Warnings such as "camera not side-on" appear bottom-left. |
| `wallballs_01_frames.csv` | One row per frame: `frame, time_s, rep_count, current_rep, phase, knee_angle, hip_angle, last_rep_verdict, last_rep_failed, proper_so_far, not_proper_so_far`. `phase` is `descent`, `drive` or `idle`. |
| `wallballs_01_reps.csv` | One row per rep: `rep, start_s, bottom_s, end_s, duration_s, descent_s, drive_s, min_knee_angle, verdict, failed_rules, advisories, not_assessable`, plus `<rule>_status` and `<rule>_value` columns for every rule. |
| `wallballs_01_verdicts.json` | Everything: exercise detection, camera-view check, warnings, per-rep detail with each rule's measured value and explanation. |
| `wallballs_01_keypoints.npz` | Cached MediaPipe output, array shape `(frames, 33, 4)` = x, y, z, visibility. Frames with no detection are NaN. |

The terminal also prints a table:

```
Exercise detection: wall_balls (confidence 1.0)
  - 5 squat cycles found (need >= 3)
  ...
  #  time(s) verdict         failed / not assessable
  1      0.7 proper          -   [n/a: target_hit, catch_after_bounce]
  3      3.7 not_proper      squat_depth
```

---

## How it works

```
video --> [track.py] MediaPipe Pose, every frame --> keypoints cache (.npz)
      --> [segment.py] clean + smooth knee/hip angle, find reps
      --> [detect.py] is this wall balls?
      --> [rules.py] judge each rep: pass / fail / not assessable
      --> [render.py] annotated video + CSV/JSON labels
```

1. **Track** (`track.py`): MediaPipe Pose (model complexity 2, the most accurate, since processing is offline) on every frame. Output is cached so rules can be tuned without re-tracking.
2. **Clean** (`segment.py`): keypoints below the visibility threshold are treated as missing and linearly interpolated. If more than 40% of a joint is missing, the clip is reported as unusable instead of guessed. The camera-facing leg is chosen automatically by mean visibility.
3. **Segment reps** (`segment.py`): the signal is the smoothed **knee angle** (hip-knee-ankle). Each squat bottom is a valley that must dip below 130° and be at least 35° deeper than its surroundings, at least 0.8 s apart. The rep boundaries are the most upright (highest-angle) frames on either side of the valley. Using peaks as boundaries means a rep is still counted when the athlete does not fully straighten, and `full_extension` then flags it instead of two reps silently merging.
4. **Detect the exercise** (`detect.py`): see below.
5. **Judge** (`rules.py`): each rule returns `pass`, `fail` or `not_assessable` with the measured value.
6. **Render** (`render.py`): overlay video and labels.

### Exercise detection

The wall-ball signature is checked with four rules:

| Check | Passes when | Separates wall balls from |
|---|---|---|
| Enough cycles | at least 3 squat cycles | random movement, single reps |
| Steady tempo | cycle length 1 to 5 s, variation (std/mean) at most 0.5 | irregular motion |
| Hands overhead | wrist rises above the shoulder by at least 0.25 torso-lengths in at least 60% of cycles | plain air squats |
| Legs in sync | median left/right knee angle difference at most 25° | alternating-leg lunges |

Labels: `wall_balls`, `squats_no_throw`, `not_wall_balls`, `uncertain`, `unknown`, with a confidence score and notes. Checks that cannot be evaluated (for example, the far leg is hidden) are skipped and reported as skipped. Verdicts are only produced for `wall_balls` or `uncertain`, unless `--force` is used.

### Rep verdict logic

- `not_proper`: at least one **non-advisory** rule failed. The failed rules are listed.
- `not_assessable`: nothing failed, but a **core** rule (squat depth) could not be judged (for example, front-on camera).
- `proper`: no failures and all core rules were judged and passed.
- **Advisory** rules (coaching tips) are shown as `[tip: ...]` and never change the verdict.

Each rep is split into phases: top → **descent** → bottom → **drive/throw** → top, and the timings are saved.

---

## Rules, sources and thresholds

All values below live in `config.py` under `RULES`. **Sources are secondary (coaching and equipment sites); I could not read the official rulebook directly. Check every row marked VERIFY against the current-season HYROX rulebook before relying on it.**

| Rule | What it checks | Threshold | Type | Source and reasoning |
|---|---|---|---|---|
| `start_standing` (rep 1 only) | Standing tall with hips and knees extended before the first squat | knee and hip angle at least 165° | Counts | HYROX rule as described by Centr: start standing tall, ball in both hands. VERIFY. 165° leaves about 15° of slack under a locked leg for tracker noise. Needs the clip to include the standing start. |
| `squat_depth` (core) | Hip crease below the top of the knee at the bottom | hip joint centre may be at most 0.08 thigh-lengths above the knee joint centre; ±0.04 is flagged "borderline" | Counts | HYROX standard, summarised by RitFit (other guides say "hips below knees / below parallel"). VERIFY. MediaPipe returns joint **centres**, not the hip crease or the top of the kneecap. The top of the knee is about 0.1 thigh-lengths above the knee centre, so 0.08 is a starting guess to be tuned against a human judge. Needs a side-on view. |
| `full_extension` | Legs and hips extended at the top of each rep | knee and hip angle at least 165° | Counts | **My assumption.** The rulebook, as far as I found, states a standing requirement only at the start and after a floor drop. VERIFY. |
| `throw_reach` | Hands go up like a throw | wrist at least 0.25 torso-lengths above the shoulder during the drive | Counts | **Heuristic proxy** for "ball thrown to the target": pose cannot see the ball. A throw to a 2.7 to 3.0 m target sends the hands well above shoulder height. This is not proof of a legal throw. |
| `arm_only_throw` | Legs drive before the arms | knee angle at least 110° at the moment the wrists pass the shoulder | **Advisory** | Technique guidance (Centr, Red Bull): drive through the legs. Not a no-rep rule. Needs a side-on view. |
| `ball_at_chest_bottom` | Ball held at chest height at the bottom | wrists between 0.0 and 0.8 of the way from shoulder to hip | **Advisory** | Technique guidance (RitFit, PureGym). Wrists are often occluded by the ball, hence advisory. |
| `target_hit` | Ball strikes the target | none | Always **not assessable** | HYROX standard: ball must hit the target (men 3.0 m, women 2.7 m; VERIFY). Ball contact is not observable from body keypoints. Carried into Part 2 (wrist IMU for release timing; confirmation needs ball/target tracking). |
| `catch_after_bounce` | Ball not caught after a bounce and carried straight into a rep | none | Always **not assessable** | Rule described by Centr and RoxLyfe (after a floor drop, stand tall before the next squat). Needs ball tracking. Carried into Part 2. |



### Other thresholds (also in `config.py`)

| Setting | Value | Purpose |
|---|---|---|
| `SIGNAL.min_visibility` | 0.5 | below this a keypoint is treated as missing |
| `SIGNAL.smooth_window_s` | 0.25 s (Savitzky-Golay, order 2) | remove jitter, keep the squat shape |
| `SIGNAL.max_missing_frac` | 0.4 | beyond this a joint is unusable |
| `VIEW.side_on_max_ratio` | 0.35 | shoulder separation / torso length; at most this means side-on |
| `SEGMENTATION.valley_max_angle` | 130° | squat bottom must go below this |
| `SEGMENTATION.min_prominence` | 35° | valley must be this much deeper than its surroundings |
| `SEGMENTATION.min_rep_s` | 0.8 s | minimum time between reps |

---

## Design decisions

- **Rules in one place.** Every threshold is in `config.py` with its source and rationale, as the brief requires. Rule code contains no numbers.
- **Cache keypoints.** Tracking is the slow part; rule tuning should not need it.
- **Knee angle as the rep signal.** It is robust to camera distance and to the athlete's height, unlike raw pixel heights, and it has a clean valley per squat.
- **Peak-based rep boundaries.** Counting by valleys (not by "return to standing") keeps fatigued, shallow-lockout reps from being merged or lost.
- **Honesty over guessing.** A rule that cannot be judged reliably returns `not_assessable` with a reason: front-on camera, low-confidence joints, hidden wrist, or no way to see the ball. A rep whose depth cannot be judged is never marked proper.
- **Advisory rules.** Coaching guidance that is not a judge's no-rep criterion is reported but cannot fail a rep.
- **Exercise detection gate.** The system refuses to judge a clip that does not look like wall balls, instead of producing confident nonsense.
- **MediaPipe Pose.** The suggested default; it is free, runs offline on CPU, and gives 33 keypoints with per-point visibility, which the not-assessable logic relies on.

---

## Assumptions

- One athlete in frame; static camera; the full body visible for the whole clip.
- The camera is roughly **side-on**. Depth and extension are only judged when the shoulder-to-torso ratio says so.
- Clips are trimmed to a continuous run of roughly 5 to 15 reps and start from the standing position (needed for `start_standing`).
- The athlete throws with both hands, so the wrist on the camera-facing side is representative.
- Thresholds are initial guesses documented in `config.py`, to be tuned on real clips.

---

## Known limitations

- **Not validated on real clips yet.** Only synthetic keypoints have been tested. Accuracy against a human judge is unknown. Fill in the results table below once measured.
- **Depth is approximate.** Joint centres stand in for the hip crease and the top of the knee, so near-threshold reps are uncertain (these are flagged "borderline").
- **Single camera.** Depth and extension are not judged from front-on or oblique views. Knee cave and left/right asymmetry cannot be assessed from a side view, and the far leg is often occluded.
- **No ball tracking.** `target_hit` and `catch_after_bounce` are never judged. `throw_reach` only proves the hands went up, not that the ball hit the target.
- **Wrist occlusion.** The ball often hides the wrists, which makes `throw_reach`, `arm_only_throw` and `ball_at_chest_bottom` unreliable; they return not assessable when the wrist is low confidence.
- **Frame rate.** At 30 fps one frame is about 33 ms, so fine timing differences are blurred.
- **Rep 1 start.** `start_standing` is only judged on rep 1, and only if the clip starts standing.
- **Race footage.** Crowds, other athletes, camera pans and odd angles will break tracking. Use static, side-on clips. Poor tracking is reported through warnings and not assessable verdicts, not hidden.
- **Detection thresholds.** The exercise-detection numbers are untuned guesses and may misclassify unusual clips; `--force` is the escape hatch.
- **Not built:** session summary, interactive web app, other stations.

---



---

## Tuning guide

1. Run a clip and watch the annotated video next to `*_reps.csv`.
2. If reps are missed or double-counted: change `SEGMENTATION` (`min_prominence`, `valley_max_angle`, `min_rep_s`).
3. If depth verdicts disagree with a human: adjust `RULES["squat_depth"]["tolerance_thigh"]`. Look at `squat_depth_value` in `*_reps.csv` for each rep, and place the threshold between your judged-good and judged-bad reps.
4. If the clip is wrongly labelled not wall balls: check the notes printed under "Exercise detection" and adjust `DETECTION`.
5. Rerun (cached keypoints are reused). Write what you changed and why in your notes.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'track'` | Run from the project folder, and make sure all `.py` files are in the same folder. |
| `No module named mediapipe` | `pip install mediapipe`. On Python 3.13 use 3.11 or 3.12. |
| `module 'mediapipe' has no attribute 'solutions'` | Newer MediaPipe versions dropped the legacy API: `pip install "mediapipe==0.10.14"`. |
| `Cannot open <video>` | Check the path. Re-encode if needed: `ffmpeg -i in.mov -c:v libx264 out.mp4`. |
| Annotated video will not play in a browser | Install `ffmpeg` and rerun; the pipeline then writes H.264. Without ffmpeg the file is `mp4v`, which VLC and most desktop players handle. |
| `No reps detected` | Check the camera angle and that the whole body is in frame. If so, lower `SEGMENTATION.min_prominence`. |
| Verdicts skipped, "does not look like wall balls" | Read the detection notes. Use `--force` to judge anyway. |
| Everything is `not_assessable` | The camera is probably not side-on. Use a side-on clip. |

---

## Project layout

```
hyrox_formcheck/
├── config.py          # ALL thresholds, with source + rationale per rule
├── track.py           # MediaPipe Pose on every frame, cache keypoints
├── segment.py         # signal cleaning, side choice, rep segmentation, leg asymmetry
├── detect.py          # "is this wall balls?" rule-based detection
├── rules.py           # per-rep rules, verdict logic, phase timings
├── render.py          # annotated video, frames.csv, reps.csv
├── run_pipeline.py    # command-line entry point
├── test_synthetic.py  # sanity test with fake keypoints (no video needed)
├── requirements.txt
└── README.md
```

---


