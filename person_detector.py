import logging
import threading
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

log = logging.getLogger(__name__)


@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: int = 0
    motion_score: float = 0.0
    is_fighting: bool = False
    is_victim: bool = False
    is_attacker: bool = False
    fight_partner_id: Optional[int] = None


    @property
    def int_box(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_prone(self) -> bool:
        return self.width >= (self.height * 1.25) and self.width >= 45.0 and self.height >= 25.0

    def check_is_prone(self, frame_w: int = 0, frame_h: int = 0) -> bool:
        if self.width < (self.height * 1.30) or self.width < 55.0 or self.height < 25.0:
            return False

        if frame_w > 0 and frame_h > 0:
            box_area = self.width * self.height
            frame_area = frame_w * frame_h

            if box_area > (frame_area * 0.18):
                return False

            if self.width > (frame_w * 0.45) or self.height > (frame_h * 0.45):
                return False

            if self.y2 < (frame_h * 0.38):
                return False

            if self.y1 < (frame_h * 0.10):
                return False

        return True


def _nms_boxes(boxes: List[PersonBox], iou_threshold: float = 0.60) -> List[PersonBox]:
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        rem = []
        for b in boxes:
            ix1 = max(best.x1, b.x1)
            iy1 = max(best.y1, b.y1)
            ix2 = min(best.x2, b.x2)
            iy2 = min(best.y2, b.y2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = best.area + b.area - inter
            iou = inter / union if union > 0 else 0.0
            min_area = min(best.area, b.area)
            iom = inter / min_area if min_area > 0 else 0.0

            if iou < iou_threshold and iom < 0.82:
                rem.append(b)
        boxes = rem

    return keep


class SimpleTracker:

    def __init__(self, max_disappeared: int = 8, smoothing_factor: float = 0.82):
        self.next_id = 1
        self.tracks: Dict[int, PersonBox] = {}
        self.disappeared: Dict[int, int] = {}
        self.smoothing = smoothing_factor
        self.max_disappeared = max_disappeared

    def update(self, detected_boxes: List[PersonBox]) -> List[PersonBox]:
        if not detected_boxes:
            for tid in list(self.tracks.keys()):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]
            active = [self.tracks[tid] for tid in self.tracks if self.disappeared[tid] <= 2]
            return _nms_boxes(active, iou_threshold=0.30)

        if not self.tracks:
            for b in detected_boxes:
                b.track_id = self.next_id
                self.tracks[self.next_id] = b
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return _nms_boxes(list(self.tracks.values()), iou_threshold=0.30)

        track_ids = list(self.tracks.keys())
        track_boxes = [self.tracks[tid] for tid in track_ids]
        det_boxes = detected_boxes

        used_tracks = set()
        used_dets = set()

        matches = []
        for t_idx, tb in enumerate(track_boxes):
            tc = tb.center
            diag = math.hypot(tb.width, tb.height)
            for d_idx, db in enumerate(det_boxes):
                dc = db.center
                ix1 = max(tb.x1, db.x1)
                iy1 = max(tb.y1, db.y1)
                ix2 = min(tb.x2, db.x2)
                iy2 = min(tb.y2, db.y2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = tb.area + db.area - inter
                iou = inter / union if union > 0 else 0.0

                dist = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
                if iou > 0.05:
                    cost = 1.0 - iou
                    matches.append((cost, t_idx, d_idx))
                elif dist <= max(260.0, diag * 2.2):
                    cost = 1.0 + (dist / max(100.0, diag))
                    matches.append((cost, t_idx, d_idx))

        matches.sort(key=lambda x: x[0])

        for cost, t_idx, d_idx in matches:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            tid = track_ids[t_idx]

            old_b = self.tracks[tid]
            new_b = det_boxes[d_idx]
            s = self.smoothing
            smooth_box = PersonBox(
                x1=old_b.x1 * (1 - s) + new_b.x1 * s,
                y1=old_b.y1 * (1 - s) + new_b.y1 * s,
                x2=old_b.x2 * (1 - s) + new_b.x2 * s,
                y2=old_b.y2 * (1 - s) + new_b.y2 * s,
                confidence=new_b.confidence,
                track_id=tid,
                motion_score=old_b.motion_score,
            )
            self.tracks[tid] = smooth_box
            self.disappeared[tid] = 0
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

        for d_idx, b in enumerate(det_boxes):
            if d_idx not in used_dets:
                b.track_id = self.next_id
                self.tracks[self.next_id] = b
                self.disappeared[self.next_id] = 0
                self.next_id += 1

        for t_idx, tid in enumerate(track_ids):
            if t_idx not in used_tracks:
                self.disappeared[tid] += 1
                if self.disappeared[tid] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.disappeared[tid]

        active_tracks = [self.tracks[tid] for tid in list(self.tracks.keys()) if self.disappeared.get(tid, 0) <= 2]
        return _nms_boxes(active_tracks, iou_threshold=0.30)

    def reset(self):
        self.tracks.clear()
        self.disappeared.clear()
        self.next_id = 1


class PersonDetector:

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.22,
        detect_every_n: int = 2,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.detect_every_n = detect_every_n
        self.iou_threshold = iou_threshold

        self._device = self._pick_device(device)
        self._model = None
        self._frame_count = 0
        self._scene_gen = 0
        self._lock = threading.Lock()
        self._cached_boxes: List[PersonBox] = []
        self._running = False
        self._first_detect = True
        self.tracker = SimpleTracker(max_disappeared=8, smoothing_factor=0.82)

        self._load_model()


    @staticmethod
    def _pick_device(pref: str) -> str:
        if pref != "auto":
            return pref
        try:
            import torch
            if torch.cuda.is_available():
                log.info("🚀 [PersonDetector] Kích hoạt GPU (CUDA) tăng tốc!")
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        try:
            from ultralytics import YOLO
            log.info(f"Đang nạp mô hình YOLO: {self.model_name} ({self._device})...")
            self._model = YOLO(self.model_name)
            log.info("✅ YOLOv8 Detector sẵn sàng!")
        except Exception as e:
            log.error(f"Lỗi nạp YOLO: {e}")
            raise

    def _detect_worker(self, frame: np.ndarray, gen: int):
        try:
            results = self._model(
                frame,
                classes=[self.PERSON_CLASS_ID],
                conf=self.confidence,
                imgsz=384,
                verbose=False,
                device=self._device,
            )
            if gen != self._scene_gen:
                return

            raw_boxes = []
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) != self.PERSON_CLASS_ID:
                        continue
                    coords = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    raw_boxes.append(
                        PersonBox(
                            x1=float(coords[0]),
                            y1=float(coords[1]),
                            x2=float(coords[2]),
                            y2=float(coords[3]),
                            confidence=conf,
                        )
                    )

            cleaned = _nms_boxes(raw_boxes, self.iou_threshold)
            with self._lock:
                if gen == self._scene_gen:
                    self._cached_boxes = self.tracker.update(cleaned)
        except Exception as e:
            log.warning(f"Lỗi detection worker: {e}")
        finally:
            self._running = False

    def reset_tracker(self):
        with self._lock:
            self._scene_gen += 1
            self.tracker.reset()
            self._cached_boxes.clear()
        self._first_detect = True

    def detect_async(self, frame: np.ndarray):
        self._frame_count += 1

        if self._first_detect:
            self._first_detect = False
        else:
            if self._frame_count % self.detect_every_n != 0:
                return

        if self._running:
            return

        self._running = True
        gen = self._scene_gen
        t = threading.Thread(
            target=self._detect_worker,
            args=(frame.copy(), gen),
            daemon=True,
        )
        t.start()


    @property
    def boxes(self) -> List[PersonBox]:
        with self._lock:
            return [
                PersonBox(
                    x1=b.x1, y1=b.y1, x2=b.x2, y2=b.y2,
                    confidence=b.confidence, track_id=b.track_id,
                    motion_score=b.motion_score, is_fighting=b.is_fighting,
                    is_victim=b.is_victim, is_attacker=b.is_attacker,
                    fight_partner_id=b.fight_partner_id,
                )
                for b in self._cached_boxes
            ]

    @staticmethod
    def analyze_fighting_clusters(
        boxes: List[PersonBox],
        orig_w: int = 0,
        orig_h: int = 0,
        motion_threshold: float = 2.8,
        is_global_violence: bool = False,
        ai_conf: float = 0.0,
    ) -> List[PersonBox]:
        if not boxes:
            return boxes

        for b in boxes:
            b.is_fighting = False
            b.is_victim = False
            b.is_attacker = False
            b.fight_partner_id = None

        n = len(boxes)
        fighting_ids = set()
        is_elevated = is_global_violence or (ai_conf >= 50.0)

        prone_candidates = [b for b in boxes if b.check_is_prone(orig_w, orig_h)]
        standing_candidates = [b for b in boxes if b not in prone_candidates]

        if prone_candidates:
            for vb in prone_candidates:
                matched_attacker = False
                for sb in standing_candidates:
                    if sb.height > (vb.height * 3.5) or sb.height > (vb.width * 2.5):
                        continue

                    diff_y2 = abs(sb.y2 - vb.y2)
                    max_ground_diff = max(45.0, sb.height * (0.42 if is_elevated else 0.28))
                    if diff_y2 > max_ground_diff:
                        continue

                    dist_x = abs(sb.center[0] - vb.center[0])
                    max_x_reach = max(vb.width * 0.75, sb.width * 1.1) * (1.3 if is_elevated else 1.0)
                    if dist_x > max_x_reach:
                        continue

                    dist = math.hypot(vb.center[0] - sb.center[0], vb.center[1] - sb.center[1])
                    max_reach = max(110.0, sb.height * (1.15 if is_elevated else 0.85))
                    if dist > max_reach:
                        continue

                    if ai_conf < 25.0 and not is_elevated:
                        continue

                    req_att_motion = 0.8 if is_elevated else 1.8
                    if is_elevated or sb.motion_score >= req_att_motion or vb.motion_score >= 1.2:
                        vb.is_victim = True
                        vb.is_fighting = True
                        sb.is_attacker = True
                        sb.is_fighting = True
                        vb.fight_partner_id = sb.track_id
                        sb.fight_partner_id = vb.track_id
                        fighting_ids.add(sb.track_id)
                        fighting_ids.add(vb.track_id)
                        matched_attacker = True
                        break

                pass

        for i in range(n):
            b1 = boxes[i]
            c1 = b1.center
            h1 = b1.height

            for j in range(i + 1, n):
                b2 = boxes[j]
                c2 = b2.center
                h2 = b2.height

                avg_h = max(30.0, (h1 + h2) / 2.0)
                avg_w = max(20.0, (b1.width + b2.width) / 2.0)

                diff_ground = abs(b1.y2 - b2.y2)
                diff_top = abs(b1.y1 - b2.y1)
                if diff_ground > avg_h * 0.22:
                    continue
                if ai_conf < 25.0 and not is_elevated and not is_global_violence:
                    if diff_ground > avg_h * 0.13 or diff_top > avg_h * 0.16:
                        continue

                x_overlap = max(0.0, min(b1.x2, b2.x2) - max(b1.x1, b2.x1))
                box_gap = max(0.0, max(b1.x1, b2.x1) - min(b1.x2, b2.x2))

                min_motion = min(b1.motion_score, b2.motion_score)
                pair_motion = max(b1.motion_score, b2.motion_score)

                if x_overlap > 0:
                    overlap_ratio = x_overlap / avg_w
                    if ai_conf < 25.0 and not is_elevated and not is_global_violence:
                        if overlap_ratio < 0.20:
                            continue
                        if min_motion < 1.3 or pair_motion < 2.3 or (min_motion + pair_motion < 3.8):
                            continue
                    if pair_motion < 2.2 and min_motion < 1.3 and ai_conf < 35.0 and not is_elevated:
                        continue
                    if (min_motion >= 1.2 and pair_motion >= 2.0 and (min_motion + pair_motion >= 3.8)) or (is_elevated and pair_motion >= 1.6) or (ai_conf >= 50.0 and pair_motion >= 1.8):
                        b1.is_fighting = True
                        b2.is_fighting = True
                        b1.fight_partner_id = b2.track_id
                        b2.fight_partner_id = b1.track_id
                        fighting_ids.add(b1.track_id)
                        fighting_ids.add(b2.track_id)

                else:
                    if box_gap > avg_h * 0.12:
                        continue
                    if (ai_conf >= 55.0 or is_elevated) and (pair_motion >= 2.5 and min_motion >= 1.0):
                        b1.is_fighting = True
                        b2.is_fighting = True
                        b1.fight_partner_id = b2.track_id
                        b2.fight_partner_id = b1.track_id
                        fighting_ids.add(b1.track_id)
                        fighting_ids.add(b2.track_id)

        if not fighting_ids and n == 1 and is_global_violence:
            b = boxes[0]
            ar = b.width / max(1.0, b.height)
            if ar >= 0.70 and b.motion_score >= 4.2 and ai_conf >= 85.0:
                b.is_fighting = True
                fighting_ids.add(b.track_id)

        return boxes
