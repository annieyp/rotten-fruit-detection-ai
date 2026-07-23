"""Fruit pose estimation for grasping: given a YOLO26 detection and an ArUco marker of
known physical size placed on the same flat surface as the fruit, estimate each
fruit's real-world position (on that surface), in-plane orientation, and width.

Works on any camera at any distance/angle -- there's no fixed working-distance
requirement and no depth sensor needed. The marker gives per-frame scale: its known
physical size vs. its detected pixel corners let us build a homography from
undistorted image pixels to real-world (X, Y) millimeters on the surface the marker
sits on. This assumes the fruit is roughly coplanar with the marker (both flat on the
same table/tray, say) -- height above that surface isn't measured, since a single
camera with no depth sensing can't recover it from one image.

Print an ArUco marker (cv2.aruco.DICT_4X4_50) at a known physical size and place it
flat on the same surface as the fruit, in view of the camera, before running this.
Needs OpenCV >= 4.7 for the cv2.aruco.ArucoDetector API.

Usage:
    python3 pose.py --weights "/Users/ayp/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Annie’s MacBook Air/Rotten Fruit Detection/best.pt" --marker-size-mm 50 --camera-index 1
"""
import argparse
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from calibration import load_calibration

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class FruitPose:
    class_name: str
    confidence: float
    position_mm: tuple  # (x, y) on the marker's surface; marker's top-left corner = (0, 0)
    orientation_deg: float  # rotation within that surface plane
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


def _find_marker_homography(undistorted, marker_size_mm, aruco_dict=ARUCO_DICT):
    """Detects the ArUco marker and returns (image_corners, homography), where
    homography maps undistorted image pixels to real-world (X, Y) mm on the marker's
    plane. Returns (None, None) if no marker was found."""
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None, None

    # detected corner order is top-left, top-right, bottom-right, bottom-left
    image_corners = corners[0].reshape(4, 2).astype(np.float32)
    world_corners = np.array(
        [[0, 0], [marker_size_mm, 0], [marker_size_mm, marker_size_mm], [0, marker_size_mm]],
        dtype=np.float32,
    )
    homography, _ = cv2.findHomography(image_corners, world_corners)
    return image_corners, homography


def _pixels_to_plane(homography, points_px):
    """points_px: (N, 2) pixel coordinates. Returns (N, 2) real-world mm on the plane."""
    points = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(points, homography)
    return mapped.reshape(-1, 2)


def estimate_poses(frame, model, camera_matrix, dist_coeffs, marker_size_mm, confidence=0.3, draw=False):
    """frame: a BGR image containing both the fruit and the reference ArUco marker.
    Returns one FruitPose per detection above `confidence`, or an empty list (with a
    printed warning) if the marker isn't visible in this frame. If draw=True, also
    returns a debug frame annotated with the marker outline, YOLO boxes, and each
    fruit's measured rotated rect -- meant for a live preview, not for measurement."""
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
    debug_frame = undistorted.copy() if draw else None

    marker_corners, homography = _find_marker_homography(undistorted, marker_size_mm)
    if homography is None:
        if draw:
            cv2.putText(
                debug_frame, "No ArUco marker found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
            return [], debug_frame
        print("No ArUco marker found in frame -- can't establish real-world scale, skipping.")
        return []

    if draw:
        cv2.polylines(debug_frame, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)

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
        # pixel space) so perspective distortion doesn't skew the angle
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
            label = f"{pose.class_name} {pose.width_mm:.0f}x{pose.length_mm:.0f}mm {pose.orientation_deg:.0f}deg"
            cv2.putText(
                debug_frame, label, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2
            )

    return (poses, debug_frame) if draw else poses


def capture_frame_picamera2():
    """Grabs one BGR frame from the Pi Camera Module 3 via picamera2 (the libcamera-
    based library -- cv2.VideoCapture doesn't reliably reach CSI cameras on Raspberry
    Pi OS). Install with `sudo apt install -y python3-picamera2`; if you're in a
    virtualenv, create it with `--system-site-packages` so it can see the apt package.
    """
    from picamera2 import Picamera2

    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    frame_rgb = picam2.capture_array()
    picam2.stop()
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def capture_frame_webcam(camera_index=1):
    """Works with any regular USB/built-in webcam (e.g. testing on a laptop) via
    OpenCV's standard VideoCapture. Use --picamera on the Pi itself instead, since the
    Camera Module 3's CSI connection needs picamera2/libcamera, not VideoCapture."""
    cap = cv2.VideoCapture(camera_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read from camera index {camera_index}")
    return frame


def iter_frames_webcam(camera_index=1):
    """Yields frames continuously from a webcam, keeping the device open across
    reads (unlike capture_frame_webcam, which is one-shot) -- for a live preview."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Could not read from camera index {camera_index}")
            yield frame
    finally:
        cap.release()


def iter_frames_picamera2():
    """Yields frames continuously from the Pi Camera Module via picamera2 -- for a
    live preview. See capture_frame_picamera2 for the apt-install note."""
    from picamera2 import Picamera2

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()
    try:
        while True:
            yield cv2.cvtColor(picam2.capture_array(), cv2.COLOR_RGB2BGR)
    finally:
        picam2.stop()


def run_live(frame_source, model, camera_matrix, dist_coeffs, marker_size_mm, confidence=0.3):
    """Shows a live window with the camera feed, the detected ArUco marker outline,
    and each fruit's measured rotated box/orientation overlaid, updating every frame.
    Press 'q' in the window to quit."""
    print("Live preview running -- press 'q' in the window to quit.")
    for frame in frame_source:
        poses, debug_frame = estimate_poses(
            frame, model, camera_matrix, dist_coeffs, marker_size_mm, confidence, draw=True
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
    parser.add_argument("--calibration", default="camera_calibration.json")
    parser.add_argument(
        "--marker-size-mm", type=float, required=True, help="physical side length of the printed ArUco marker"
    )
    parser.add_argument("--image", help="test on a single image file instead of a live camera")
    parser.add_argument("--camera-index", type=int, default=0, help="webcam index (laptop testing)")
    parser.add_argument(
        "--picamera", action="store_true", help="capture via picamera2 (Pi Camera Module) instead of --camera-index"
    )
    parser.add_argument(
        "--live", action="store_true", help="show a live preview window instead of a single capture+print"
    )
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    model = YOLO(args.weights)

    if args.live:
        frame_source = iter_frames_picamera2() if args.picamera else iter_frames_webcam(args.camera_index)
        run_live(frame_source, model, camera_matrix, dist_coeffs, args.marker_size_mm)
    else:
        if args.image:
            frame = cv2.imread(args.image)
        elif args.picamera:
            frame = capture_frame_picamera2()
        else:
            frame = capture_frame_webcam(args.camera_index)

        for pose in estimate_poses(frame, model, camera_matrix, dist_coeffs, args.marker_size_mm):
            print(pose)
