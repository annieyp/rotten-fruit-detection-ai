"""Interactive photo capture for calibration, reusing pose.py's exact camera-opening
code (iter_frames_webcam / iter_frames_picamera2) -- guarantees these photos come from
the same camera and resolution pose.py will actually use later, avoiding the mismatch
that happens when calibration photos come from a different app (e.g. Photo Booth).

Usage:
    python3 capture_photos.py --output-dir chessboard_photos --camera-index 1

Press SPACE to save the current frame, 'q' to quit. Take ~15-20 photos of your
chessboard at different angles/distances/positions before quitting (see
calibration.py's docstring for what makes a good set).

"""
import argparse
from pathlib import Path

import cv2

from pose import iter_frames_picamera2, iter_frames_webcam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--camera-index", type=int, default=1, help="webcam index (laptop testing)")
    parser.add_argument("--capture-width", type=int, help="request this capture width from the webcam")
    parser.add_argument("--capture-height", type=int, help="request this capture height from the webcam")
    parser.add_argument(
        "--picamera", action="store_true", help="capture via picamera2 (Pi Camera Module) instead of --camera-index"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_size = (args.capture_width, args.capture_height) if args.capture_width and args.capture_height else None
    frame_source = (
        iter_frames_picamera2() if args.picamera else iter_frames_webcam(args.camera_index, capture_size)
    )

    print("SPACE to save a photo, 'q' to quit. Aim for 15-20 photos at different angles/distances.")
    count = 0
    for frame in frame_source:
        cv2.imshow("capture_photos -- SPACE to save, q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            path = output_dir / f"photo_{count:02d}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Saved {path} ({frame.shape[1]}x{frame.shape[0]})")
            count += 1
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    print(f"Saved {count} photos to {output_dir}")
