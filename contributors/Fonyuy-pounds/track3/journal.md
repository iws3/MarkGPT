# Track 3 — Journal

**Author:** Patrick
**Track:** Track 3 — Tracking & Rule Engine
**Project:** Classroom Monitoring & Alarm System
**Built on:** "MediaPipe From First Principles" — Part 3 (Gita, SEED ML)

---

## What I was asked to do

Track 3 takes the raw per-frame detections from Track 2 (Faith) and turns
them into a sustained judgment about a specific person — giving each
student a stable ID across frames, and computing a debounced suspicion
score for them.

Track 2 returns a list of poses per frame, but that list is not stable.
The same physical person can be index 0 in one frame and index 2 in the
next. So we cannot say "Student #2 has been leaning for 15 frames" until
we have a stable Student #2.

That is what I built.

---

## What I built

| File | What it does |
|---|---|
| `track3/trackers/centroid_tracker.py` | Assigns a stable ID to each person across frames by matching nearest hip-midpoint centroids. |
| `track3/trackers/suspicion_tracker.py` | Combines head yaw, torso lean, and object-near-hand into one weighted score per student, with debouncing. |
| `track3/adapter.py` | Converts Faith's `structured_output` into the shape Track 3 expects. |
| `track3/main.py` | End-to-end entry point — runs the full pipeline on a webcam or a video file. |
| `track3/diag.py` | Diagnostic tool — prints ID creation over time, for checking tracker stability. |
| `track3/mock/fake_detections.py` | Fake detections for testing without a camera. |
| `track3/__init__.py` and package markers | So imports work. |
| `.gitignore` | Extended to ignore `venv/`, `*.task`, `test_videos/`. |

---

## How it works

**CentroidTracker** — for each frame:

1. Take the hip-midpoint of each detected person.
2. Match it to the closest tracked person from the last frame.
3. If within `max_distance=0.15`, it is the same person — keep the ID.
4. Otherwise, it is a new person — new ID.
5. If a person is not seen for `max_missed_frames=20` frames, forget them.

All matching is in normalized coordinates (0–1), so it works at any
resolution.

**StudentSuspicionTracker** — for each tracked student, each frame:

- Head yaw > 30° → +0.3
- Torso lean > 20° → +0.3
- Object near hand → +0.4

If the score stays at or above 0.6 for 15 frames in a row, the tracker
returns `should_alert = True` for that student.

Weighted sum, not AND or OR: AND misses real cases, OR is too twitchy,
and a weighted sum with debounce is the middle ground.

---

## What I tested and what I got

Ran the full pipeline on a real multi-person video (1080p music clip
with scene cuts):

| Metric | Result |
|---|---|
| Frames processed | 1312 |
| Average FPS | 13.2 |
| Unique IDs created | 34 |

**What worked well:**

- When a single person is in frame for a long stretch, the ID stays
  the same for 200+ consecutive frames. Matching logic is correct.
- 13.2 FPS on CPU is above the 10 FPS target.
- End-to-end pipeline runs without errors.

**Known issue:**

- The test clip has many scene cuts and short-lived background faces.
  Some 1–2 frame detections get their own ID before disappearing.
  That is why there are 34 IDs even though the clip has about 10 real
  people.

---

## What is not done yet

- **Object detection.** Faith's pipeline does not run an object detector
  yet, so `object_near_hand` is always `False` for now.
- **Confirmation debounce for new IDs.** Planned as a follow-up — require
  a new detection to appear for a few frames before giving it a permanent
  ID. This will kill most of the phantom IDs.
- **Tuning the weights.** The current numbers (0.3 / 0.3 / 0.4, threshold
  0.6, 15 frames) are starting points. They need to be tuned on real
  classroom footage.
- **Alarm sound and dashboard.** That is Track 4 (Wohking).

---

## Decisions worth noting

**Normalized coordinates, not pixels.** The same distance threshold
works at any resolution, which matters because the phone stream
(Track 1) may change resolution mid-session.

**Hip-midpoint as the centroid.** The hips are the most stable part of
the body — less affected by gestures and head turns.

**Per-student debounce counters.** Each student's behavior is
independent. One student crossing the threshold should not affect
another.

**Object-near-hand weighted highest (0.4).** It is the strongest
signal — the difference between stretching and using a phone under
the desk. Head yaw and torso lean are supporting evidence.

**15 frames of debounce.** At ~13 FPS that is about one second. Enough
to filter brief noise, short enough to catch real sustained behavior.

---

## Interface contract

Track 2 → Track 3, one entry per detected person per frame:

```python
{
    "centroid": (x_norm, y_norm),   # hip-midpoint, normalized [0, 1]
    "head_yaw_deg": float | None,
    "torso_lean_deg": float | None,
    "object_near_hand": bool,
    "timestamp_ms": int,
}