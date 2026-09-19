import cv2
import logging
import threading
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


class MotionAnalyzer:

    def __init__(
        self,
        fighting_threshold: float = 3.0,
        scale: float = 0.25,
        skip_frames: int = 2,
    ):
        self.fighting_threshold = fighting_threshold
        self.scale = scale
        self.skip_frames = skip_frames

        self._prev_gray: Optional[np.ndarray] = None
        self._magnitude: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._frame_count = 0
        self._orig_size: Optional[tuple] = None

    def update_async(self, frame: np.ndarray):
        self._frame_count += 1
        if self._frame_count % self.skip_frames != 0:
            return
        if self._running:
            return

        self._running = True
        self._orig_size = (frame.shape[0], frame.shape[1])
        t = threading.Thread(
            target=self._worker,
            args=(frame.copy(),),
            daemon=True,
        )
        t.start()

    def _worker(self, frame: np.ndarray):
        try:
            small = cv2.resize(
                frame, (0, 0), fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            with self._lock:
                prev = self._prev_gray

            if prev is not None and prev.shape == gray.shape:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, gray, None,
                    pyr_scale=0.5, levels=2, winsize=11,
                    iterations=2, poly_n=5, poly_sigma=1.1,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

                with self._lock:
                    self._magnitude = mag
                    self._prev_gray = gray
            else:
                with self._lock:
                    self._prev_gray = gray

        except Exception as e:
            log.warning(f"[MotionAnalyzer] Lỗi: {e}")
        finally:
            self._running = False

    def reset(self):
        with self._lock:
            self._prev_gray = None
            self._magnitude = None
            self._frame_count = 0

    def reset_scene(self, frame: np.ndarray):
        try:
            h, w = frame.shape[:2]
            small = cv2.resize(
                frame, (0, 0), fx=self.scale, fy=self.scale,
                interpolation=cv2.INTER_LINEAR,
            )
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            with self._lock:
                self._prev_gray = gray
                self._magnitude = np.zeros(gray.shape, dtype=np.float32)
                self._orig_size = (h, w)
        except Exception as e:
            log.warning(f"[MotionAnalyzer] Lỗi reset_scene: {e}")

    def get_global_motion(self) -> tuple:
        with self._lock:
            mag = self._magnitude
        if mag is None or mag.size == 0:
            return 0.0, 0.0, 0.0
        return float(np.mean(mag)), float(np.percentile(mag, 95)), float(np.max(mag))

    def score_boxes(self, boxes: list) -> list:
        with self._lock:
            mag = self._magnitude

        if mag is None or not boxes:
            return boxes

        h_m, w_m = mag.shape
        sc = self.scale
        for box in boxes:
            x1 = max(0, min(int(round(box.x1 * sc)), w_m - 1))
            y1 = max(0, min(int(round(box.y1 * sc)), h_m - 1))
            x2 = max(x1 + 1, min(int(round(box.x2 * sc)), w_m))
            y2 = max(y1 + 1, min(int(round(box.y2 * sc)), h_m))
            region = mag[y1:y2, x1:x2]
            box.motion_score = (
                float(np.percentile(region, 75)) if region.size > 0 else 0.0
            )
        return boxes

    def is_fighting(self, box) -> bool:
        return box.motion_score >= self.fighting_threshold

    def get_flow_heatmap(self, frame_shape: tuple) -> Optional[np.ndarray]:
        with self._lock:
            mag = self._magnitude
        if mag is None:
            return None
        h, w = frame_shape[:2]
        mag_r = cv2.resize(mag, (w, h))
        mn    = cv2.normalize(mag_r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hmap  = cv2.applyColorMap(mn, cv2.COLORMAP_INFERNO)
        mask  = mn > int(self.fighting_threshold * 8)
        result = np.zeros_like(hmap)
        result[mask] = hmap[mask]
        return result
