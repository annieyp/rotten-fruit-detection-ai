"""Camera calibration via a chessboard pattern (OpenCV). Run this once per physical
camera/lens; it saves intrinsics to disk for pose.py to load later.

This doesn't need to run on the Raspberry Pi itself -- take ~15-20 photos of a
chessboard at different angles/distances (on the Pi: `rpicam-still -o img1.jpg`,
repeated, or `rpicam-still -t 0` for a live preview to line up each shot), copy the
JPGs to any computer, and run this there. Easier to debug with --show on a machine
with a display than on the Pi itself.

Usage:
    python3 calibration.py --images-dir chessboard_photos --pattern-size 7 6 --square-size-mm 15

Note:
    For pattern size, 7 x 6 is the inner corners, so the actual board should b 8 x 7
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np


def calibrate(images_dir, pattern_size=(7, 6), square_size_mm=25.0, show=False):
    """pattern_size = (inner corners per row, inner corners per column) -- e.g. a
    standard 8x7-square chessboard has a 7x6 pattern of *inner* corners, not squares.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm  # real-world units, so calibration output is in mm

    objpoints, imgpoints = [], []
    # set() to dedupe: on case-insensitive filesystems (e.g. default macOS), "*.jpg"
    # and "*.JPG" match the same files, which would otherwise double-count every photo.
    image_paths = sorted(
        {
            p
            for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png")
            for p in glob.glob(str(Path(images_dir) / pattern))
        }
    )
    if not image_paths:
        raise ValueError(f"No .jpg/.jpeg/.png images found in {images_dir}")

    gray_shape = None
    for path in image_paths:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if not found:
            print(f"Skipping {path}: chessboard not found")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        if show:
            cv2.drawChessboardCorners(img, pattern_size, corners, found)
            cv2.imshow("corners", img)
            cv2.waitKey(300)

    if show:
        cv2.destroyAllWindows()

    if len(objpoints) < 10:
        print(
            f"Warning: only {len(objpoints)} usable images -- calibration accuracy "
            "will suffer. Aim for 15-20+ covering different angles/positions."
        )

    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_shape, None, None
    )

    mean_error = 0.0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        # cornerSubPix and projectPoints don't always agree on point-array shape
        # ((N, 2) vs (N, 1, 2)) across OpenCV versions -- normalize both before cv2.norm.
        actual = imgpoints[i].reshape(-1, 2)
        projected = projected.reshape(-1, 2)
        mean_error += cv2.norm(actual, projected, cv2.NORM_L2) / len(projected)
    print(f"Mean reprojection error (want well under 1.0): {mean_error / len(objpoints):.4f}")

    return camera_matrix, dist_coeffs


def save_calibration(path, camera_matrix, dist_coeffs):
    Path(path).write_text(
        json.dumps({"camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist()})
    )


def load_calibration(path):
    data = json.loads(Path(path).read_text())
    return np.array(data["camera_matrix"]), np.array(data["dist_coeffs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--pattern-size", nargs=2, type=int, default=[7, 6])
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    parser.add_argument("--output", default="camera_calibration.json")
    parser.add_argument("--show", action="store_true", help="display detected corners as it goes")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = calibrate(
        args.images_dir, tuple(args.pattern_size), args.square_size_mm, show=args.show
    )
    save_calibration(args.output, camera_matrix, dist_coeffs)
    print(f"Saved calibration to {args.output}")
