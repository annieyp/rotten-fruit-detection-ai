"""One-time extrinsic calibration for a fixed, tilted camera mount.

Run this exactly once per physical camera position/tilt: place a printed ArUco marker
(cv2.aruco.DICT_4X4_50) of known size flat on the picking surface, in view of the
camera, then press SPACE to capture and save the resulting homography. After that,
remove the marker permanently -- pose.py loads this saved homography instead of
detecting a marker live, so the camera's position and tilt must stay fixed from this
point on. Redo this calibration any time the mount gets bumped or repositioned.

This only works because the camera's tilt/position is FIXED: the homography this
computes is only valid for exactly the pose the camera was in when you pressed SPACE.

Usage:
    python3 calibrate_extrinsics.py --calibration camera_calibration.json --marker-size-mm 50 --camera-index 1
"""
import argparse

import cv2

from calibration import find_marker_homography, load_calibration, save_extrinsic_calibration
from camera import iter_frames_picamera2, iter_frames_webcam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", default="camera_calibration.json", help="intrinsics from calibration.py")
    parser.add_argument(
        "--marker-size-mm", type=float, required=True, help="physical side length of the printed ArUco marker"
    )
    parser.add_argument("--output", default="extrinsic_calibration.json")
    parser.add_argument("--camera-index", type=int, default=1, help="webcam index (laptop testing)")
    parser.add_argument("--capture-width", type=int, help="request this capture width from the webcam")
    parser.add_argument("--capture-height", type=int, help="request this capture height from the webcam")
    parser.add_argument(
        "--picamera", action="store_true", help="capture via picamera2 (Pi Camera Module) instead of --camera-index"
    )
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    capture_size = (args.capture_width, args.capture_height) if args.capture_width and args.capture_height else None
    frame_source = (
        iter_frames_picamera2() if args.picamera else iter_frames_webcam(args.camera_index, capture_size)
    )

    print("Place the marker on the picking surface. SPACE to capture once it's detected (green outline), 'q' to quit.")
    for frame in frame_source:
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        image_corners, homography = find_marker_homography(undistorted, args.marker_size_mm)

        preview = undistorted.copy()
        if image_corners is not None:
            cv2.polylines(preview, [image_corners.astype("int32")], True, (0, 255, 0), 2)
        else:
            cv2.putText(preview, "No marker found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("calibrate_extrinsics -- SPACE to capture, q to quit", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and homography is not None:
            save_extrinsic_calibration(args.output, homography)
            print(f"Saved extrinsic calibration to {args.output}. Remove the marker now -- setup is done.")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()
