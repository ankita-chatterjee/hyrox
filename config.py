"""
ALL tunable numbers live here. Nothing in the other modules hard-codes a threshold.

Every rule has `source` (where it comes from) and `rationale` (why this threshold).
`advisory: True` means the rule is reported (coaching) but does NOT change the rep verdict.
Items marked VERIFY must be checked against the current-season HYROX rulebook before you cite them.
"""

# ---------------- Tracking (MediaPipe Pose) ----------------
TRACKER = dict(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ---------------- Signal cleaning ----------------
SIGNAL = dict(
    min_visibility=0.5,       # keypoints below this are treated as missing, then interpolated
    smooth_window_s=0.25,     # Savitzky-Golay window (sec)
    smooth_polyorder=2,
    max_missing_frac=0.4,     # joint unusable if more than this fraction is missing
)

# ---------------- Camera view check ----------------
VIEW = dict(side_on_max_ratio=0.35)   # |L-R shoulder x| / torso length; ~0 side-on, large front-on

# ---------------- Rep segmentation ----------------
# Signal: knee angle (hip-knee-ankle) of the better-visible leg. ~170 standing, ~60-90 deep squat.
SEGMENTATION = dict(valley_max_angle=130, min_prominence=35, min_rep_s=0.8)

# ---------------- Exercise detection: "is this clip wall balls?" ----------------
# Wall-ball signature = repeated squat cycles + hands going overhead each cycle + both legs moving together.
# Distinguishes from: air squats (no overhead reach), lunges (legs out of sync), burpees (few/irregular squat cycles).
DETECTION = dict(
    min_reps=3,
    overhead_reach_torso=0.25,   # wrist above shoulder by this many torso-lengths counts as "hands overhead"
    min_overhead_frac=0.6,       # fraction of cycles that must show overhead reach
    min_period_s=1.0,            # wall-ball cycle length bounds (elite ~1.5-2 s, fatigued up to ~4 s)
    max_period_s=5.0,
    max_period_cv=0.5,           # std/mean of cycle length; very irregular => not a steady station set
    max_leg_asym_deg=25,         # median |left knee angle - right knee angle|; large => alternating legs (lunges)
    min_leg_vis_frac=0.3,        # need both legs visible this often to use the leg-sync check
)

# ---------------- Reference only (not used by code) ----------------
# Sources disagree on ball weights (e.g. Open Men 6 kg vs 9 kg Pro; Open Women 4 kg). VERIFY against the rulebook.
# Targets: men 3.0 m (10 ft), women 2.7 m (9 ft).
WALL_BALL_REFERENCE = dict(reps=100, target_m=dict(men=3.0, women=2.7))

# ---------------- Rules ----------------
RULES = {
    "start_standing": dict(
        enabled=True, core=False, advisory=False,
        min_knee_angle=165, min_hip_angle=165, min_visibility=0.5,
        source="HYROX rule (via Centr): racer must start standing tall, hips and knees extended, "
               "holding the ball with both hands. Applies to rep 1 (and after a floor drop). VERIFY wording.",
        rationale="165 deg leaves ~15 deg slack under a fully locked leg for tracker noise. "
                  "Only judged on rep 1: the clip must be trimmed to include the standing start.",
    ),
    "squat_depth": dict(
        enabled=True, core=True, advisory=False,
        tolerance_thigh=0.08,     # hip joint centre may sit this fraction of thigh length above knee centre
        borderline_band=0.04,
        min_visibility=0.5, needs_side_view=True,
        source="HYROX standard: hip crease must drop below the top of the knee (RitFit); other guides "
               "say 'hips below knees / below parallel' (BoxRox, Red Bull). VERIFY in current-season rulebook.",
        rationale="MediaPipe gives joint CENTRES, not the hip crease or top of the kneecap. The top of the knee "
                  "sits ~one knee-radius (~0.1 thigh length) above the knee joint centre, so 0.08 is a starting "
                  "guess. Tune against a human judge on your clips.",
    ),
    "full_extension": dict(
        enabled=True, core=False, advisory=False,
        min_knee_angle=165, min_hip_angle=165, min_visibility=0.5, needs_side_view=True,
        source="ASSUMPTION: the drive/throw should finish with the legs extended. Rulebook only states a "
               "standing requirement at the start / after a floor drop. VERIFY.",
        rationale="165 deg, same slack as start_standing.",
    ),
    "throw_reach": dict(
        enabled=True, core=False, advisory=False,
        wrist_above_shoulder_torso=0.25, min_visibility=0.5, needs_side_view=False,
        source="HEURISTIC PROXY for 'ball thrown to the target' (pose cannot see the ball): hands must go "
               "up and out like a throw.",
        rationale="A throw to a 2.7-3.0 m target sends the hands well above shoulder height. Not proof of a legal throw.",
    ),
    "arm_only_throw": dict(
        enabled=True, core=False, advisory=True,
        min_knee_angle_when_arms_rise=110, min_visibility=0.5, needs_side_view=True,
        source="Technique guidance (Centr, Red Bull): drive through the legs, arms only guide the ball. "
               "Coaching rule, not a no-rep rule.",
        rationale="If the wrists pass shoulder height while the knee is still deeply bent (<110 deg), "
                  "the throw started before the legs drove. Advisory only.",
    ),
    "ball_at_chest_bottom": dict(
        enabled=True, core=False, advisory=True,
        # wrist position along shoulder->hip line at the bottom (0 = shoulder, 1 = hip)
        min_frac=0.0, max_frac=0.8, min_visibility=0.5, needs_side_view=False,
        source="Technique guidance (RitFit, PureGym): hold the ball at chest height, elbows tucked. Coaching rule.",
        rationale="Wrists should sit between shoulder and hip height at the bottom; far outside means the ball "
                  "is held low or out in front. Advisory only: ball and wrists are often occluded.",
    ),
    "target_hit": dict(
        enabled=True, core=False, advisory=False, always_not_assessable=True,
        reason="Ball contact with the target is not observable from body keypoints. Carried into Part 2 "
               "(IMU on wrist/forearm for release timing; real fix needs ball/target tracking).",
        source="HYROX standard: each throw must strike the centre of the designated target "
               "(men 3.0 m, women 2.7 m). VERIFY target heights per division.",
    ),
    "catch_after_bounce": dict(
        enabled=True, core=False, advisory=False, always_not_assessable=True,
        reason="Whether the ball bounced before the catch needs ball tracking; pose cannot tell.",
        source="HYROX rule (Centr, RoxLyfe): after a floor drop the racer must stand tall before the next squat; "
               "cannot catch after the bounce and continue straight into a rep.",
    ),
}
