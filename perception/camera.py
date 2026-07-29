"""Camera frame capture, shared by pose.py, calibrate_extrinsics.py, and
capture_photos.py. Deliberately has no dependency on ultralytics/YOLO -- keeping this
separate means calibration/capture scripts don't need PyTorch installed at all."""
import cv2


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


def _apply_capture_resolution(cap, capture_size):
    """Requests a specific (width, height) from the camera, then warns if the driver
    didn't actually honor it -- camera_matrix's pixel-coordinate assumptions (cx, cy,
    fx, fy) are only valid at the resolution calibration.py was run at, so a silent
    mismatch here would make cv2.undistort warp frames incorrectly."""
    if capture_size is None:
        return
    width, height = capture_size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actual = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual != (float(width), float(height)):
        print(
            f"Warning: requested {width}x{height} capture but the camera is giving "
            f"{actual[0]:.0f}x{actual[1]:.0f} instead -- your calibration won't match "
            "this resolution. Recalibrate at this actual resolution, or find a "
            "resolution the camera will actually honor."
        )


def capture_frame_webcam(camera_index=1, capture_size=None):
    """Works with any regular USB/built-in webcam (e.g. testing on a laptop) via
    OpenCV's standard VideoCapture. Use --picamera on the Pi itself instead, since the
    Camera Module 3's CSI connection needs picamera2/libcamera, not VideoCapture.
    capture_size: optional (width, height) to request -- should match whatever
    resolution calibration.py was run at, since camera_matrix is resolution-specific.
    """
    cap = cv2.VideoCapture(camera_index)
    _apply_capture_resolution(cap, capture_size)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read from camera index {camera_index}")
    return frame


def iter_frames_webcam(camera_index=1, capture_size=None):
    """Yields frames continuously from a webcam, keeping the device open across
    reads (unlike capture_frame_webcam, which is one-shot) -- for a live preview.
    capture_size: see capture_frame_webcam."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")
    _apply_capture_resolution(cap, capture_size)
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
