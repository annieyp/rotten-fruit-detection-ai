"""Fruit pose estimation for grasping: given a YOLO26 detection and a camera mounted
at a FIXED position/tilt above the picking surface, estimate each fruit's real-world
position, in-plane orientation, and width.

Unlike a straight-down mount, a tilted camera means distance-to-surface isn't constant
across the frame, so a simple pixel/focal-length formula isn't enough -- this instead
loads a homography (pixel -> real-world mm on the picking surface) computed ONCE by
calibrate_extrinsics.py, via a marker placed at your actual fixed mount/tilt and then
removed. This is only valid as long as the camera's physical position and tilt never
change after that calibration -- if the mount gets bumped, redo calibrate_extrinsics.py.

orientation_deg is relative to whatever axes calibrate_extrinsics.py's marker defined
when it was calibrated (its own printed top edge = 0deg), not the image frame or any
physical marker still present during operation -- see calibrate_extrinsics.py.

Usage:
    source .venv/bin/activate
    python3 pose.py --weights "/Users/ayp/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Annie’s MacBook Air/Rotten Fruit Detection/best.pt" --extrinsic-calibration extrinsic_calibration.json --camera-index 0 --live
"""

import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from calibration import load_calibration, load_extrinsic_calibration
from camera import (
    capture_frame_picamera2,
    capture_frame_webcam,
    iter_frames_picamera2,
    iter_frames_webcam,
)


@dataclass
class FruitPose:
    class_name: str
    confidence: float
    position_mm: tuple  # (x, y) on the calibrated surface plane
    orientation_deg: float  # rotation within that plane
    width_mm: float  # shorter side of the fruit's rotated bounding rect
    length_mm: float  # longer side


def _segment_largest_contour(crop):
    """Adaptive-polarity Otsu threshold + largest external contour within a detection
    crop. Plain THRESH_BINARY always marks the *brighter* region as foreground, which
    silently breaks whenever the fruit is the darker of the two (a dark eggplant on a
    light tray, say) -- it would pick up the background instead. Since a YOLO box
    always leaves at least a little background visible near its edges, this samples
    the crop's outer border pixels and picks whichever threshold direction treats that
    border as background. Assumes the fruit contrasts reasonably against the
    background at all -- swap in HSV color thresholding if it doesn't."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_value, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]])
    if np.mean(border) >= thresh_value:
        mask = cv2.bitwise_not(mask)  # border was mislabeled as foreground; flip it

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _pixels_to_plane(homography, points_px):
    """points_px: (N, 2) pixel coordinates. Returns (N, 2) real-world mm on the plane."""
    points = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, homography)
    return mapped.reshape(-1, 2)


def estimate_poses(frame, model, camera_matrix, dist_coeffs, homography, confidence=0.3, draw=False):
    """frame: a BGR image. homography: from calibrate_extrinsics.py (via
    load_extrinsic_calibration), NOT detected live from this frame. Returns one
    FruitPose per detection above `confidence`. If draw=True, also returns a debug
    frame annotated with YOLO boxes and each fruit's measured rotated rect -- meant
    for a live preview, not for measurement."""
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
    debug_frame = undistorted.copy() if draw else None

    result = model.predict(undistorted, verbose=False)[0]

    poses = []
    for box in result.boxes:
        if float(box.conf[0]) < confidence:
            continue

        xmin, ymin, xmax, ymax = (int(v) for v in box.xyxy[0].tolist())
        crop = undistorted[ymin:ymax, xmin:xmax]
        contour = _segment_largest_contour(crop)
        if contour is None:
            if draw:
                cv2.rectangle(debug_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            continue

        rect = cv2.minAreaRect(contour)
        box_pts = cv2.boxPoints(rect)  # 4 corners, pixel space, relative to the crop
        box_pts[:, 0] += xmin  # shift back to full-frame pixels
        box_pts[:, 1] += ymin

        world_pts = _pixels_to_plane(homography, box_pts)  # 4 corners, real-world mm
        side_a = np.linalg.norm(world_pts[1] - world_pts[0])
        side_b = np.linalg.norm(world_pts[2] - world_pts[1])
        width_mm, length_mm = sorted([side_a, side_b])

        center_mm = world_pts.mean(axis=0)
        # orientation of the longer side, measured in the real-world plane (not raw
        # pixel space) so perspective distortion from the camera's tilt doesn't skew it
        long_edge = world_pts[2] - world_pts[1] if side_b >= side_a else world_pts[1] - world_pts[0]
        orientation_deg = float(np.degrees(np.arctan2(long_edge[1], long_edge[0])))

        pose = FruitPose(
            class_name=result.names[int(box.cls[0])],
            confidence=float(box.conf[0]),
            position_mm=(float(center_mm[0]), float(center_mm[1])),
            orientation_deg=orientation_deg,
            width_mm=float(width_mm),
            length_mm=float(length_mm),
        )
        poses.append(pose)

        if draw:
            cv2.drawContours(debug_frame, [box_pts.astype(np.int32)], 0, (255, 128, 0), 2)
            label = f"{pose.class_name} {pose.confidence:.2f} {pose.width_mm:.0f}x{pose.length_mm:.0f}mm {pose.orientation_deg:.0f}deg"
            cv2.putText(
                debug_frame, label, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2
            )

    return (poses, debug_frame) if draw else poses


def run_live(frame_source, model, camera_matrix, dist_coeffs, homography, confidence=0.3):
    """Shows a live window with the camera feed and each fruit's measured rotated
    box/orientation overlaid, printing every detection on every frame as soon as it's
    computed -- no batching/averaging across frames. Fruit here is moving (conveyor
    belt), so there's no window of frames where it holds still long enough to average
    meaningfully; whatever needs to fuse readings over time (e.g. to predict where a
    still-moving fruit will be) belongs downstream, where the belt's known velocity
    lives, not here. Press 'q' in the window to quit."""
    print("Live preview running -- press 'q' in the window to quit.")
    for frame in frame_source:
        poses, debug_frame = estimate_poses(
            frame, model, camera_matrix, dist_coeffs, homography, confidence, draw=True
        )
        cv2.imshow("pose.py live preview", debug_frame)
        for pose in poses:
            print(pose)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="path to trained YOLO26 .pt weights")
    parser.add_argument("--calibration", default="camera_calibration.json", help="intrinsics from calibration.py")
    parser.add_argument(
        "--extrinsic-calibration",
        default="extrinsic_calibration.json",
        help="homography from calibrate_extrinsics.py (one-time setup, not detected live)",
    )
    parser.add_argument("--image", help="test on a single image file instead of a live camera")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index (laptop testing)")
    parser.add_argument(
        "--picamera", action="store_true", help="capture via picamera2 (Pi Camera Module) instead of --camera-index"
    )
    parser.add_argument(
        "--live", action="store_true", help="show a live preview window instead of a single capture+print"
    )
    parser.add_argument(
        "--capture-width", type=int, help="request this capture width from the webcam (must match calibration.py's resolution)"
    )
    parser.add_argument(
        "--capture-height", type=int, help="request this capture height from the webcam (must match calibration.py's resolution)"
    )
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    homography = load_extrinsic_calibration(args.extrinsic_calibration)
    model = YOLO(args.weights)
    capture_size = (args.capture_width, args.capture_height) if args.capture_width and args.capture_height else None

    if args.live:
        frame_source = (
            iter_frames_picamera2()
            if args.picamera
            else iter_frames_webcam(args.camera_index, capture_size)
        )
        run_live(frame_source, model, camera_matrix, dist_coeffs, homography)
    else:
        if args.image:
            frame = cv2.imread(args.image)
        elif args.picamera:
            frame = capture_frame_picamera2()
        else:
            frame = capture_frame_webcam(args.camera_index, capture_size)

        for pose in estimate_poses(frame, model, camera_matrix, dist_coeffs, homography):
            print(pose)
