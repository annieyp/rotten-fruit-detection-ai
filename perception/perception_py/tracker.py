"""Tracks fruit across frames on a conveyor belt moving at a known, constant speed --
unlike detection alone (pose.estimate_poses), which treats every frame's detections as
unrelated, this gives each physical fruit a persistent ID as it moves through the
camera's view, and predicts where it'll be next based on the belt's known velocity.

This works BECAUSE the belt's motion is simple: constant speed, constant direction.
Predicting where a tracked fruit will be next frame is just "last known position +
belt_velocity * elapsed_time" -- no need for a general Kalman filter that has to guess
at unknown/changing motion, which is what real trackers (SORT, DeepSORT) need for
unpredictable subjects like pedestrians or vehicles.

belt_direction must be measured for your actual setup: it's a (dx, dy) vector in the
same plane calibrate_extrinsics.py established (the marker's plane), pointing the
direction fruit actually travels -- it gets normalized internally, so any nonzero
vector in the right direction works. Determine it empirically: run pose.py --live,
watch how position_mm changes for a real fruit crossing the frame, and use that
direction.
"""
import math
import time
from dataclasses import dataclass


@dataclass
class Track:
    id: int
    class_name: str
    position_mm: tuple  # last actually-observed position, not a live prediction
    orientation_deg: float
    width_mm: float
    length_mm: float
    confidence: float
    last_seen: float  # time.monotonic() of the last matched detection
    frames_since_seen: int = 0


class FruitTracker:
    def __init__(self, belt_speed_mm_s, belt_direction, max_match_distance_mm=40.0, max_missed_frames=5):
        """belt_speed_mm_s: constant belt speed, e.g. 25 m/min -> pass 25_000 / 60.
        belt_direction: (dx, dy), any nonzero vector pointing the direction fruit
        travels (normalized internally). max_missed_frames: how many consecutive
        frames a track can go unmatched before it's dropped (fruit has left the
        belt's visible area)."""
        norm = math.hypot(*belt_direction)
        self._velocity_mm_s = (
            belt_direction[0] / norm * belt_speed_mm_s,
            belt_direction[1] / norm * belt_speed_mm_s,
        )
        self._max_match_distance_mm = max_match_distance_mm
        self._max_missed_frames = max_missed_frames
        self._tracks = {}
        self._next_id = 0

    def _predicted_position(self, track, now):
        elapsed = now - track.last_seen
        vx, vy = self._velocity_mm_s
        return (track.position_mm[0] + vx * elapsed, track.position_mm[1] + vy * elapsed)

    def update(self, detections, now=None):
        """detections: this frame's list of FruitPose (from pose.estimate_poses).
        Returns the current list of active Tracks, including ones not matched this
        frame but still within max_missed_frames."""
        if now is None:
            now = time.monotonic()

        unmatched_detections = list(detections)

        for track in self._tracks.values():
            predicted = self._predicted_position(track, now)

            best_detection, best_dist = None, None
            for detection in unmatched_detections:
                if detection.class_name != track.class_name:
                    continue
                dist = math.hypot(
                    predicted[0] - detection.position_mm[0], predicted[1] - detection.position_mm[1]
                )
                if dist <= self._max_match_distance_mm and (best_dist is None or dist < best_dist):
                    best_detection, best_dist = detection, dist

            if best_detection is not None:
                track.position_mm = best_detection.position_mm
                track.orientation_deg = best_detection.orientation_deg
                track.width_mm = best_detection.width_mm
                track.length_mm = best_detection.length_mm
                track.confidence = best_detection.confidence
                track.last_seen = now
                track.frames_since_seen = 0
                unmatched_detections.remove(best_detection)
            else:
                track.frames_since_seen += 1

        for detection in unmatched_detections:
            track = Track(
                id=self._next_id,
                class_name=detection.class_name,
                position_mm=detection.position_mm,
                orientation_deg=detection.orientation_deg,
                width_mm=detection.width_mm,
                length_mm=detection.length_mm,
                confidence=detection.confidence,
                last_seen=now,
            )
            self._tracks[track.id] = track
            self._next_id += 1

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.frames_since_seen <= self._max_missed_frames
        }

        return list(self._tracks.values())

    def predicted_position_now(self, track, now=None):
        """Where a track's fruit currently is, extrapolated forward from its last
        actual detection using the belt's known velocity -- for a grasp-planning
        consumer that needs a position estimate between detections, not just at them."""
        if now is None:
            now = time.monotonic()
        return self._predicted_position(track, now)
