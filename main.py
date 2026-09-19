from __future__ import annotations
import os, sys, cv2, time, logging, argparse, threading, math
import numpy as np
from datetime import datetime
from collections import deque
from typing import Optional, List, Tuple

import yaml
import tensorflow as tf

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

from notifier import DiscordNotifier, TelegramNotifier
from logger_module import DetectionLogger
from person_detector import PersonDetector, PersonBox
from optical_flow import MotionAnalyzer

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("system.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("main")


load_model = tf.keras.models.load_model
MobileNetV2 = tf.keras.applications.MobileNetV2
layers = tf.keras.layers

_orig_input = layers.InputLayer.__init__
def _patch_input(self, *a, **kw):
    kw.pop("optional", None)
    if "batch_shape" in kw:
        kw["batch_input_shape"] = kw.pop("batch_shape")
    _orig_input(self, *a, **kw)
layers.InputLayer.__init__ = _patch_input

_orig_dense = layers.Dense.__init__
def _patch_dense(self, *a, **kw):
    kw.pop("quantization_config", None)
    _orig_dense(self, *a, **kw)
layers.Dense.__init__ = _patch_dense


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        log.error(f"Không tìm thấy cấu hình: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class FrameGrabber:

    def __init__(self, source, is_stream: bool = False):
        self.source = source
        self.is_stream = is_stream

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Không thể kết nối nguồn video: {source}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = raw_fps if (0 < raw_fps < 240) else 25.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._stopped = False
        self._eof = False

        if self.is_stream:
            self._latest_frame: Optional[np.ndarray] = None
            self._has_frame = False
            self._lock = threading.Lock()
            self._thread = threading.Thread(target=self._stream_loop, daemon=True)
            self._thread.start()
            for _ in range(80):
                with self._lock:
                    if self._has_frame:
                        break
                time.sleep(0.05)

    def _stream_loop(self):
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                log.warning("[FrameGrabber] Mất tín hiệu luồng RTSP, kết nối lại sau 2s...")
                time.sleep(2.0)
                self._cap.open(self.source)
                continue
            with self._lock:
                self._latest_frame = frame
                self._has_frame = True

    def get_latest(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if not self._has_frame or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def read_next(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._eof:
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            self._eof = True
            return False, None
        return True, frame

    def stop(self):
        self._stopped = True
        self._cap.release()


class ViolenceDetector:

    CLASSES = ["NonViolence", "Violence"]

    def __init__(self, seq_model_path: str, input_size: int = 96, seq_len: int = 10):
        log.info("Dang khoi dong AI Engine: MobileNetV2 + Temporal LSTM...")
        self.seq_model = load_model(seq_model_path)
        self.feature_extractor = MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=(input_size, input_size, 3), pooling="avg",
        )
        self.input_size = input_size
        self.seq_len = seq_len

        feat_ext = self.feature_extractor
        seq_m = self.seq_model

        @tf.function
        def _fast_extract(x):
            return feat_ext(x, training=False)

        @tf.function
        def _fast_predict(x):
            return seq_m(x, training=False)

        self._fast_extract = _fast_extract
        self._fast_predict = _fast_predict

        dummy_img = tf.zeros((1, input_size, input_size, 3), dtype=tf.float32)
        dummy_seq = tf.zeros((1, seq_len, 1280), dtype=tf.float32)
        _ = self._fast_extract(dummy_img)
        _ = self._fast_predict(dummy_seq)

        self.alpha_up   = 0.28
        self.alpha_down = 0.08
        self._threat_hold = 0

        self.feature_buffer = deque(maxlen=seq_len)
        self._raw_prob = 0.0
        self._smooth_prob = 0.0
        self._label = "Analyzing..."
        self._is_violence_stable = False
        self._violence_counter = 0

        self._lock = threading.Lock()
        self._inferring = False
        log.info("AI Engine san sang hoat dong! (Graph Compiled 15ms/infer)")

    @property
    def label(self) -> str:
        with self._lock:
            return self._label

    @property
    def confidence(self) -> float:
        with self._lock:
            return self._smooth_prob * 100.0

    @property
    def is_violence(self) -> bool:
        with self._lock:
            return self._is_violence_stable

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        m = min(h, w)
        crop = frame_bgr[(h - m) // 2 : (h - m) // 2 + m, (w - m) // 2 : (w - m) // 2 + m]
        small = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def _worker_infer(self, rgb_frame: np.ndarray):
        try:
            inp = tf.constant(np.expand_dims(rgb_frame, axis=0), dtype=tf.float32)
            feat = self._fast_extract(inp).numpy()[0]

            with self._lock:
                self.feature_buffer.append(feat)
                if len(self.feature_buffer) < self.seq_len:
                    self._inferring = False
                    return

                seq_arr = tf.constant(np.expand_dims(np.array(self.feature_buffer, dtype=np.float32), axis=0))

            preds = self._fast_predict(seq_arr).numpy()[0]
            viol_prob = float(preds[1])

            with self._lock:
                self._raw_prob = viol_prob

                if viol_prob >= self._smooth_prob:
                    alpha = self.alpha_up
                else:
                    alpha = self.alpha_down

                self._smooth_prob = alpha * viol_prob + (1.0 - alpha) * self._smooth_prob

                if self._smooth_prob >= 0.65:
                    self._threat_hold = 50
                    self._is_violence_stable = True
                    self._label = "Violence"
                elif self._threat_hold > 0 and self._smooth_prob >= 0.48:
                    self._threat_hold -= 1
                    self._is_violence_stable = True
                    self._label = "Violence"
                else:
                    self._threat_hold = 0
                    if self._smooth_prob >= 0.45:
                        self._is_violence_stable = False
                        self._label = "Elevated"
                    else:
                        self._is_violence_stable = False
                        self._label = "NonViolence"

        except Exception as e:
            log.error(f"[ViolenceDetector] Loi xu ly suy luan: {e}")
        finally:
            with self._lock:
                self._inferring = False

    def on_scene_cut(self, frame_bgr: np.ndarray):
        try:
            rgb = self._preprocess_frame(frame_bgr)
            inp = tf.constant(np.expand_dims(rgb, axis=0), dtype=tf.float32)
            feat = self._fast_extract(inp).numpy()[0]

            with self._lock:
                self.feature_buffer.clear()
                self.feature_buffer.append(feat)
                self._raw_prob = 0.0
                self._smooth_prob = 0.0
                self._threat_hold = 0
                self._is_violence_stable = False
                self._label = "NonViolence"

            log.info("[SceneCut] Chuyen canh sach se! Buffer duoc lam moi hoan toan.")
        except Exception as e:
            log.error(f"[ViolenceDetector] Loi on_scene_cut: {e}")

    def cancel_threat(self):
        with self._lock:
            self._threat_hold = 0
            self._smooth_prob = min(self._smooth_prob, 0.20)
            self._raw_prob = min(self._raw_prob, 0.20)
            self._is_violence_stable = False
            self._label = "NonViolence"

    def fast_safe_cooldown(self):
        with self._lock:
            if self._threat_hold > 0:
                return
            if self._smooth_prob > 0.05:
                self._smooth_prob *= 0.85
                if self._smooth_prob < 0.38:
                    self._is_violence_stable = False
                    self._label = "NonViolence"

    def push_frame_async(self, frame_bgr: np.ndarray):
        with self._lock:
            if self._inferring:
                return
            self._inferring = True

        rgb = self._preprocess_frame(frame_bgr)
        t = threading.Thread(target=self._worker_infer, args=(rgb,), daemon=True)
        t.start()


class TacticalHUD:

    GRAPH_LEN = 160
    GRAPH_H = 50

    def __init__(self, show_fps=True, show_stats=True, show_graph=True):
        self.show_fps = show_fps
        self.show_stats = show_stats
        self.show_graph = show_graph

        self.fps_hist = deque(maxlen=25)
        self.conf_hist = deque(maxlen=self.GRAPH_LEN)
        self._t_prev = time.time()
        self.alert_count = 0
        self.start_time = time.time()
        self.pulse_phase = 0.0

    def tick(self):
        now = time.time()
        dt = max(now - self._t_prev, 1e-6)
        self._t_prev = now
        self.fps_hist.append(1.0 / dt)
        self.pulse_phase += dt * 6.0

    def push_conf(self, c: float):
        self.conf_hist.append(c)

    def inc_alert(self):
        self.alert_count += 1

    @property
    def fps(self) -> float:
        return float(np.mean(self.fps_hist)) if self.fps_hist else 0.0

    @property
    def uptime(self) -> str:
        e = int(time.time() - self.start_time)
        h, r = divmod(e, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def draw(
        self,
        frame: np.ndarray,
        label: str,
        conf: float,
        is_violence: bool,
        cam_label: str,
        sound_enabled: bool,
        person_count: int = 0,
        fighting_count: int = 0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        sc = w / 800.0

        f_title = max(0.40, 0.52 * sc)
        f_main = max(0.48, 0.70 * sc)
        f_sub = max(0.32, 0.42 * sc)
        header_h = max(38, int(52 * sc))

        if is_violence and conf >= 60.0:
            alpha = (math.sin(self.pulse_phase) + 1.0) / 2.0
            border_thick = max(3, int(6 * sc))
            b_col = (int(30 * (1 - alpha)), int(30 * (1 - alpha)), int(210 + 45 * alpha))
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), b_col, border_thick)

        header_slice = frame[0:header_h, 0:w]
        if header_slice.size > 0:
            dark_hdr = np.full_like(header_slice, (12, 16, 20), dtype=np.uint8)
            cv2.addWeighted(dark_hdr, 0.75, header_slice, 0.25, 0, header_slice)
        cv2.line(frame, (0, header_h), (w, header_h), (50, 70, 85), 1)

        rec_dot_color = (0, 0, 255) if int(time.time() * 2) % 2 == 0 else (100, 100, 100)
        cv2.circle(frame, (int(18 * sc), int(header_h * 0.45)), int(5 * sc), rec_dot_color, -1)

        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(
            frame,
            f"LIVE FEED  |  {cam_label.upper()}  |  {now_str}",
            (int(32 * sc), int(header_h * 0.48)),
            cv2.FONT_HERSHEY_DUPLEX,
            f_sub,
            (210, 225, 235),
            1,
            cv2.LINE_AA,
        )

        if is_violence and conf >= 60.0:
            status_text = f"CRITICAL: VIOLENCE DETECTED [{conf:.1f}%]"
            status_color = (0, 30, 240)
        elif is_violence or conf >= 45.0:
            status_text = f"ELEVATED THREAT [{conf:.1f}%]"
            status_color = (0, 180, 255)
        else:
            status_text = "NORMAL / SECURE"
            status_color = (0, 220, 90)

        cv2.putText(
            frame,
            status_text,
            (int(32 * sc), int(header_h * 0.88)),
            cv2.FONT_HERSHEY_DUPLEX,
            f_main,
            status_color,
            max(1, int(2 * sc)),
            cv2.LINE_AA,
        )

        right_x = w - int(240 * sc)
        if self.show_fps:
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (right_x, int(header_h * 0.45)),
                cv2.FONT_HERSHEY_SIMPLEX,
                f_sub,
                (0, 230, 230),
                1,
                cv2.LINE_AA,
            )

        if self.show_stats:
            person_txt = f"PERSONS: {person_count}"
            if fighting_count > 0:
                person_txt += f"  FIGHTING: {fighting_count}"
            person_color = (0, 80, 255) if fighting_count > 0 else (0, 210, 130)

            cv2.putText(
                frame,
                person_txt,
                (right_x - int(70 * sc), int(header_h * 0.66)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.28, f_sub * 0.85),
                person_color,
                1,
                cv2.LINE_AA,
            )


            alarm_dot_x = right_x - int(88 * sc)
            alarm_dot_y = int(header_h * 0.87)
            alarm_dot_r = max(4, int(5 * sc))
            if sound_enabled:
                cv2.circle(frame, (alarm_dot_x, alarm_dot_y), alarm_dot_r, (0, 220, 0), -1)
            else:
                cv2.circle(frame, (alarm_dot_x, alarm_dot_y), alarm_dot_r, (80, 80, 80), -1)
            cv2.circle(frame, (alarm_dot_x, alarm_dot_y), alarm_dot_r, (200, 210, 215), 1)

            snd_tag = "ALARM:ON" if sound_enabled else "ALARM:OFF"
            cv2.putText(
                frame,
                f"ALERTS:{self.alert_count}  UP:{self.uptime}  {snd_tag}",
                (alarm_dot_x + alarm_dot_r + int(6 * sc), alarm_dot_y + int(4 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX,
                max(0.28, f_sub * 0.85),
                (180, 195, 205),
                1,
                cv2.LINE_AA,
            )


        if self.show_graph and len(self.conf_hist) > 1:
            gw = min(w // 2, int(self.GRAPH_LEN * 2.8 * sc))
            gh = max(35, int(self.GRAPH_H * sc))
            gx, gy = int(12 * sc), h - gh - int(12 * sc)

            bg_slice = frame[gy:gy + gh, gx:gx + gw]
            if bg_slice.size > 0:
                dark_bg = np.full_like(bg_slice, (15, 18, 22), dtype=np.uint8)
                cv2.addWeighted(dark_bg, 0.70, bg_slice, 0.30, 0, bg_slice)
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (45, 60, 75), 1)

            ty = gy + int(gh * (1.0 - 0.75))
            cv2.line(frame, (gx, ty), (gx + gw, ty), (0, 140, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "75% THRESHOLD", (gx + 4, max(gy + 8, ty - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, f_sub * 0.7, (0, 140, 255), 1)

            pts = list(self.conf_hist)
            if len(pts) >= 3:
                s_pts = [pts[0]]
                for idx in range(1, len(pts) - 1):
                    s_pts.append(0.25 * pts[idx - 1] + 0.50 * pts[idx] + 0.25 * pts[idx + 1])
                s_pts.append(pts[-1])
                pts = s_pts

            step = max(1.0, gw / float(self.GRAPH_LEN))

            if len(pts) > 1:
                poly_coords = [(gx, gy + gh - 1)]
                for i in range(len(pts)):
                    px = int(gx + i * step)
                    py = gy + int(gh * (1.0 - min(100.0, pts[i]) / 100.0))
                    poly_coords.append((px, py))
                poly_coords.append((int(gx + (len(pts) - 1) * step), gy + gh - 1))

                wave_slice = frame[gy:gy + gh, gx:gx + gw]
                if wave_slice.size > 0:
                    local_coords = [(px - gx, py - gy) for (px, py) in poly_coords]
                    glow_patch = wave_slice.copy()
                    last_c = pts[-1]
                    fill_color = (0, 35, 230) if last_c >= 70.0 else ((0, 150, 240) if last_c >= 40.0 else (0, 190, 70))
                    cv2.fillPoly(glow_patch, [np.array(local_coords, dtype=np.int32)], fill_color)
                    cv2.addWeighted(glow_patch, 0.25, wave_slice, 0.75, 0, wave_slice)

            for i in range(1, len(pts)):
                x_a = int(gx + (i - 1) * step)
                x_b = int(gx + i * step)
                y_a = gy + int(gh * (1.0 - min(100.0, pts[i - 1]) / 100.0))
                y_b = gy + int(gh * (1.0 - min(100.0, pts[i]) / 100.0))
                c_line = (0, 45, 255) if pts[i] >= 70.0 else ((0, 180, 255) if pts[i] >= 40.0 else (0, 230, 90))
                cv2.line(frame, (x_a, y_a), (x_b, y_b), c_line, max(2, int(2.2 * sc)), cv2.LINE_AA)

            cv2.putText(
                frame,
                "VIOLENCE PROBABILITY WAVE",
                (gx + 4, gy + gh - int(5 * sc)),
                cv2.FONT_HERSHEY_SIMPLEX,
                f_sub * 0.75,
                (170, 185, 195),
                1,
                cv2.LINE_AA,
            )

        help_txt = "[B]: An Khung  [TAB]: Clean Feed  [SPACE]: Pause  [H]: Heatmap  [G]: Wave  [A]: Audio  [Q]: Exit"
        cv2.putText(
            frame,
            help_txt,
            (w - int(540 * sc), h - int(12 * sc)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.28, f_sub * 0.76),
            (175, 195, 205),
            1,
            cv2.LINE_AA,
        )

        return frame


def draw_tactical_boxes(
    frame: np.ndarray,
    boxes: List[PersonBox],
    orig_w: int,
    orig_h: int,
    is_global_violence: bool,
    conf: float = 0.0,
) -> np.ndarray:
    if not boxes:
        return frame

    dh, dw = frame.shape[:2]
    sx = dw / float(orig_w)
    sy = dh / float(orig_h)
    sc = dw / 800.0

    fs = max(0.32, 0.44 * sc)
    corner_len = max(12, int(20 * sc))
    c_th = max(3, int(3.5 * sc))

    mapped_boxes = []
    for b in boxes:
        x1 = int(b.x1 * sx)
        y1 = int(b.y1 * sy)
        x2 = int(b.x2 * sx)
        y2 = int(b.y2 * sy)
        mapped_boxes.append((b, x1, y1, x2, y2))

    drawn_pairs = set()
    if is_global_violence:
        for b, x1, y1, x2, y2 in mapped_boxes:
            if b.is_fighting and b.fight_partner_id:
                pair_key = tuple(sorted([b.track_id, b.fight_partner_id]))
                if pair_key not in drawn_pairs:
                    drawn_pairs.add(pair_key)
                    partner = next((item for item in mapped_boxes if item[0].track_id == b.fight_partner_id), None)
                    if partner:
                        _, px1, py1, px2, py2 = partner
                        c1 = ((x1 + x2) // 2, (y1 + y2) // 2)
                        c2 = ((px1 + px2) // 2, (py1 + py2) // 2)
                        dist_px = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                        box_h = max(y2 - y1, py2 - py1)

                        partner_box = partner[0]
                        min_box_h = min(y2 - y1, py2 - py1)
                        x_ov = max(0, min(x2, px2) - max(x1, px1))
                        if x_ov > 15 and dist_px < max(80.0, min_box_h * 0.85) and abs(y2 - py2) < (min_box_h * 0.35):
                            cv2.line(frame, c1, c2, (0, 0, 255), max(2, int(2.5 * sc)), cv2.LINE_AA)
                            mid_x = (c1[0] + c2[0]) // 2
                            mid_y = (c1[1] + c2[1]) // 2
                            cv2.putText(
                                frame,
                                "CLASHING",
                                (mid_x - int(24 * sc), mid_y - int(6 * sc)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fs * 0.85,
                                (0, 0, 255),
                                1,
                                cv2.LINE_AA,
                            )

    occupied_tags = []

    for b, x1, y1, x2, y2 in mapped_boxes:
        if b.is_victim:
            primary_col = (235, 60, 210)
            label_text = f"VICTIM #{b.track_id} [DOWN]"
            tag_bg = (160, 20, 140)
        elif b.is_attacker:
            primary_col = (0, 25, 240)
            label_text = f"ATTACKER #{b.track_id} [ACTIVE {b.motion_score:.1f}]"
            tag_bg = (0, 10, 190)
        elif b.is_fighting:
            primary_col = (0, 25, 240)
            label_text = f"TARGET #{b.track_id} [FIGHTING {b.motion_score:.1f}]"
            tag_bg = (0, 10, 190)
        elif is_global_violence:
            if b.motion_score >= 3.0:
                primary_col = (0, 25, 240)
                label_text = f"TARGET #{b.track_id} [VIOLENCE {b.motion_score:.1f}]"
                tag_bg = (0, 10, 190)
            else:
                primary_col = (0, 215, 65)
                label_text = f"BYSTANDER #{b.track_id} [SAFE]"
                tag_bg = (0, 130, 35)
        elif b.motion_score >= 3.0:
            primary_col = (0, 220, 130)
            label_text = f"PERSON #{b.track_id} [ACTIVE]"
            tag_bg = (0, 130, 70)
        else:
            primary_col = (0, 215, 65)
            label_text = f"PERSON #{b.track_id} [SECURE]"
            tag_bg = (0, 130, 35)

        cv2.rectangle(frame, (x1, y1), (x2, y2), primary_col, 1)

        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), primary_col, c_th)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), primary_col, c_th)

        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), primary_col, c_th)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), primary_col, c_th)

        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), primary_col, c_th)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), primary_col, c_th)

        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), primary_col, c_th)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), primary_col, c_th)

        dot_r = max(2, int(2.5 * sc))
        cv2.circle(frame, (x1, y1), dot_r, primary_col, -1)
        cv2.circle(frame, (x2, y1), dot_r, primary_col, -1)
        cv2.circle(frame, (x1, y2), dot_r, primary_col, -1)
        cv2.circle(frame, (x2, y2), dot_r, primary_col, -1)

        (tw, tht), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        pad = 5
        tag_h = tht + pad * 2
        lx1 = max(0, min(x1, dw - tw - pad * 2 - 8))
        lx2 = lx1 + tw + pad * 2 + 8

        if b.is_victim:
            ly1 = min(dh - tag_h, y2 + 4)
            ly2 = ly1 + tag_h
        else:
            ly1 = max(0, y1 - tag_h - 2)
            ly2 = ly1 + tag_h

        for ox1, oy1, ox2, oy2 in occupied_tags:
            if not (lx2 < ox1 or lx1 > ox2 or ly2 < oy1 or ly1 > oy2):
                ly1 = min(dh - tag_h, oy2 + 2)
                ly2 = ly1 + tag_h
                break

        occupied_tags.append((lx1, ly1, lx2, ly2))

        tag_overlay = frame[ly1:ly2, lx1:lx2]
        if tag_overlay.size > 0:
            t_bg = np.full_like(tag_overlay, tag_bg, dtype=np.uint8)
            cv2.addWeighted(tag_overlay, 0.20, t_bg, 0.80, 0, tag_overlay)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), primary_col, 1)

        cv2.rectangle(frame, (lx1, ly1), (lx1 + max(3, int(4 * sc)), ly2), (255, 255, 255), -1)

        cv2.putText(
            frame,
            label_text,
            (lx1 + pad + int(4 * sc), ly2 - pad - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            fs,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def trigger_audio_alarm():
    if HAS_WINSOUND:
        try:
            winsound.Beep(1400, 220)
        except Exception:
            pass


def save_snapshot(frame: np.ndarray, folder: str, label: str, cam: str) -> str:
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe = cam.replace(" ", "_").replace("/", "_")
    path = os.path.join(folder, f"{ts}_{safe}_{label}.jpg")
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    log.info(f"📸 [Snapshot Đã Lưu] {path}")
    return path


def build_notifiers(cfg: dict):
    discord = telegram = None
    d = cfg.get("discord", {})
    if d.get("enabled") and d.get("webhook_url"):
        discord = DiscordNotifier(d["webhook_url"], d.get("mention_role_id", ""))
        log.info("🔔 Discord Notifier: ĐÃ KÍCH HOẠT")
    t = cfg.get("telegram", {})
    if t.get("enabled") and t.get("bot_token") and t.get("chat_id"):
        telegram = TelegramNotifier(t["bot_token"], t["chat_id"])
        log.info("🔔 Telegram Notifier: ĐÃ KÍCH HOẠT")
    return discord, telegram


class SceneCutDetector:
    def __init__(self, mad_threshold: float = 26.0, corr_threshold: float = 0.60):
        self.mad_threshold = mad_threshold
        self.corr_threshold = corr_threshold
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None

    def check_cut(self, frame: np.ndarray) -> bool:
        try:
            small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)

            if self._prev_gray is None or self._prev_hist is None:
                self._prev_gray = gray
                self._prev_hist = hist
                return False

            mad = float(np.mean(cv2.absdiff(gray, self._prev_gray)))
            corr = float(cv2.compareHist(hist, self._prev_hist, cv2.HISTCMP_CORREL))

            self._prev_gray = gray
            self._prev_hist = hist

            return (mad >= self.mad_threshold) or (corr <= self.corr_threshold)
        except Exception:
            return False


def detect_active_roi(frame: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    col_mean = gray.mean(axis=0)
    row_mean = gray.mean(axis=1)
    col_std = gray.std(axis=0)
    row_std = gray.std(axis=1)

    active_cols = np.where((col_mean > 30) & (col_std > 12))[0]
    active_rows = np.where((row_mean > 30) & (row_std > 12))[0]

    if len(active_cols) > 0.4 * w and len(active_rows) > 0.4 * h:
        x1, x2 = int(active_cols[0]), int(active_cols[-1])
        y1, y2 = int(active_rows[0]), int(active_rows[-1])
        if (x1 > 0.04 * w or x2 < 0.96 * w or y1 > 0.04 * h or y2 < 0.96 * h):
            return x1, y1, x2, y2
    return 0, 0, w, h


class CameraRunner:

    def __init__(
        self,
        source,
        cfg: dict,
        detector: ViolenceDetector,
        det_logger: DetectionLogger,
        discord=None,
        telegram=None,
        cam_label: str = "Camera",
        save_output: bool = True,
        out_filename: str = "output.mp4",
    ):
        self.source = source
        self.cfg = cfg
        self.detector = detector
        self.logger = det_logger
        self.discord = discord
        self.telegram = telegram
        self.cam_label = cam_label
        self.save_output = save_output
        self.out_filename = out_filename

        disp = cfg["display"]
        self.hud = TacticalHUD(
            show_fps=disp["show_fps"],
            show_stats=disp["show_stats"],
            show_graph=disp["show_mini_graph"],
        )

        det = cfg["detection"]
        snap = cfg["snapshot"]

        self.conf_threshold = det["confidence_threshold"]
        self.cooldown = det["cooldown_seconds"]
        self.snap_enabled = snap["enabled"]
        self.snap_folder = snap["folder"]
        self._last_alert = 0.0
        self._last_beep = 0.0
        self._session_start = time.time()
        self.sound_enabled = True

        self._sc_detector = SceneCutDetector(mad_threshold=26.0, corr_threshold=0.60)


        self.ui_show_heatmap = cfg.get("motion_analysis", {}).get("show_heatmap", False)
        self.ui_show_boxes = True
        self.ui_clean_mode = False
        self.is_paused = False

        pd = cfg.get("person_detection", {})
        self.pd_enabled = pd.get("enabled", True)
        self._pd: Optional[PersonDetector] = None
        if self.pd_enabled:
            self._pd = PersonDetector(
                model_name=pd.get("model", "yolov8n.pt"),
                confidence=pd.get("confidence", 0.18),
                detect_every_n=pd.get("detect_every_n_frames", 2),
                device=pd.get("device", "auto"),
            )

        ma = cfg.get("motion_analysis", {})
        self.ma_enabled = ma.get("enabled", True)
        self.heatmap_alpha = ma.get("heatmap_alpha", 0.30)
        self.fighting_motion_threshold = ma.get("fighting_motion_threshold", 2.8)
        self._ma: Optional[MotionAnalyzer] = None
        if self.ma_enabled:
            self._ma = MotionAnalyzer(
                fighting_threshold=self.fighting_motion_threshold,
                scale=ma.get("flow_scale", 0.25),
                skip_frames=ma.get("flow_skip_frames", 2),
            )

    def _handle_alert(self, frame: np.ndarray, lbl: str, conf: float):
        now = time.time()
        if now - self._last_alert < self.cooldown:
            return

        self._last_alert = now
        self.hud.inc_alert()

        snap = ""
        if self.snap_enabled:
            snap = save_snapshot(frame, self.snap_folder, lbl, self.cam_label)

        self.logger.log_event(lbl, conf, self.cam_label, snap)

        for notif in [self.discord, self.telegram]:
            if notif:
                threading.Thread(
                    target=notif.send_alert,
                    args=(lbl, conf, frame.copy(), self.cam_label),
                    daemon=True,
                ).start()

        log.warning(f"🚨 [ALERT KÍCH HOẠT] {self.cam_label} | {lbl} ({conf:.1f}%) | Snap: {snap or 'None'}")

    def run(self):
        src = self.source
        is_stream = (
            isinstance(src, int)
            or (isinstance(src, str) and (
                src.startswith("rtsp://") or src.startswith("http://") or src.startswith("https://")
            ))
        )

        if isinstance(src, str) and not is_stream and not os.path.exists(src):
            log.error(f"Tệp video không tồn tại: {src}")
            return

        try:
            grabber = FrameGrabber(src, is_stream=is_stream)
        except Exception as e:
            log.error(f"Khởi động nguồn thất bại: {e}")
            return

        w, h = grabber.width, grabber.height
        fps = grabber.fps
        log.info(f"[{self.cam_label}] Khởi tạo thành công: {w}x{h} @ {fps:.1f} FPS | Mode: {'STREAM' if is_stream else 'FILE'}")

        rec = self.cfg["recording"]
        out = None
        if self.save_output and rec["enabled"] and w > 0 and h > 0:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_fps = fps if (0 < fps < 120) else rec["fps_fallback"]
            out = cv2.VideoWriter(self.out_filename, fourcc, out_fps, (w, h))
            log.info(f"[{self.cam_label}] Ghi video ra: {self.out_filename}")

        disp = self.cfg["display"]
        dw, dh = disp["window_width"], disp["window_height"]
        win = f"Violence Detection Pro - {self.cam_label}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, dw, dh)

        print("\n" + "=" * 76)
        print("🎯 HƯỚNG DẪN ĐIỀU KHIỂN PHÍM TẮT TRÊN CỬA SỔ VIDEO:")
        print("   👉 [B]      : ẨN / HIỆN TOÀN BỘ KHUNG BOUNDING BOX (HỘP NGƯỜI)")
        print("   👉 [TAB]/[C]: CHẾ ĐỘ CLEAN FEED (ẨN/HIỆN TOÀN BỘ GIAO DIỆN HUD)")
        print("   👉 [SPACE]  : TẠM DỪNG / TIẾP TỤC VIDEO")
        print("   👉 [H]      : BẬT / TẮT HEATMAP DÒNG QUANG HỌC")
        print("   👉 [G]      : BẬT / TẮT BIỂU ĐỒ XUNG NHỊP BẠO LỰC")
        print("   👉 [A]      : BẬT / TẮT CÒI BÁO ĐỘNG ÂM THANH")
        print("   👉 [S]      : CHỤP ẢNH BẰNG CHỨNG TỨC THÌ (SNAPSHOT)")
        print("   👉 [Q]/[ESC]: THOÁT AN TOÀN")
        print("=" * 76 + "\n")

        frame_interval = 1.0 / fps
        last_frame_t = 0.0
        processed_frames = 0
        rx1, ry1, rx2, ry2 = 0, 0, w, h
        has_roi = False
        roi_checked = False

        try:
            while True:
                if self.is_paused:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord(" "), ord("p")):
                        self.is_paused = False
                    elif key in (ord("q"), 27):
                        break
                    continue

                if is_stream:
                    ret, frame = grabber.get_latest()
                    if not ret or frame is None:
                        time.sleep(0.005)
                        continue
                else:
                    frame_start_t = time.time()
                    ret, frame = grabber.read_next()
                    if not ret:
                        log.info(f"[{self.cam_label}] Video kết thúc.")
                        break

                processed_frames += 1

                if not roi_checked:
                    rx1, ry1, rx2, ry2 = detect_active_roi(frame)
                    has_roi = (rx2 - rx1 < 0.95 * w or ry2 - ry1 < 0.95 * h)
                    roi_checked = True
                    if has_roi:
                        log.info(f"[{self.cam_label}] Phát hiện VMS ROI: ({rx1}, {ry1}) -> ({rx2}, {ry2}) | Kích thước: {rx2-rx1}x{ry2-ry1}")

                is_cut = self._sc_detector.check_cut(frame)
                if is_cut:
                    rx1, ry1, rx2, ry2 = detect_active_roi(frame)
                    has_roi = (rx2 - rx1 < 0.95 * w or ry2 - ry1 < 0.95 * h)
                    ai_frame = frame[ry1:ry2, rx1:rx2] if has_roi else frame
                    self.detector.on_scene_cut(ai_frame)
                    if self._ma:
                        self._ma.reset_scene(frame)
                    if self._pd:
                        self._pd.reset_tracker()
                        self._pd.detect_async(frame)
                else:
                    ai_frame = frame[ry1:ry2, rx1:rx2] if has_roi else frame
                    self.detector.push_frame_async(ai_frame)

                if self._ma and not is_cut:
                    self._ma.update_async(frame)

                if self.pd_enabled and self._pd:
                    self._pd.detect_async(frame)

                person_boxes = self._pd.boxes if (self.pd_enabled and self._pd) else []
                if person_boxes and self._ma:
                    person_boxes = self._ma.score_boxes(person_boxes)

                if person_boxes and self._pd:
                    person_boxes = self._pd.analyze_fighting_clusters(
                        boxes=person_boxes,
                        orig_w=w,
                        orig_h=h,
                        motion_threshold=self.fighting_motion_threshold,
                        is_global_violence=self.detector.is_violence,
                        ai_conf=self.detector.confidence,
                    )

                n_persons = len(person_boxes)
                n_fighting = sum(1 for b in person_boxes if b.is_fighting)
                has_victim = any(b.is_victim for b in person_boxes)
                has_attacker = any(b.is_attacker for b in person_boxes)

                lbl = self.detector.label
                conf = self.detector.confidence
                is_violence = self.detector.is_violence

                mean_mag, p95_mag, max_mag = self._ma.get_global_motion() if self._ma else (0.0, 0.0, 0.0)
                max_box_motion = max((b.motion_score for b in person_boxes), default=0.0)

                min_pair_dist = 999.0
                for i_idx in range(n_persons):
                    for j_idx in range(i_idx + 1, n_persons):
                        avg_h_p = (person_boxes[i_idx].height + person_boxes[j_idx].height) / 2.0
                        d_p = math.hypot(
                            person_boxes[i_idx].center[0] - person_boxes[j_idx].center[0],
                            person_boxes[i_idx].center[1] - person_boxes[j_idx].center[1]
                        ) / max(1.0, avg_h_p)
                        if d_p < min_pair_dist:
                            min_pair_dist = d_p

                is_lone_single_human = (n_persons == 1 and not (has_victim and has_attacker))

                if processed_frames <= 30:
                    if is_violence or conf > 20.0:
                        self.detector.cancel_threat()
                        is_violence = False
                        lbl = "Initializing..."
                        conf = min(conf, 20.0)

                elif n_persons == 0:
                    if is_violence or conf > 15.0:
                        self.detector.cancel_threat()
                        is_violence = False
                        lbl = "NonViolence"
                        conf = min(conf, 15.0)

                elif is_lone_single_human:
                    if is_violence or conf > 18.0:
                        self.detector.cancel_threat()
                        is_violence = False
                        lbl = "NonViolence"
                        conf = min(conf, 18.0)

                elif n_persons >= 3 and n_fighting == 0 and not has_victim and not has_attacker and (min_pair_dist > 1.25 or max_box_motion < 1.8):
                    if is_violence or conf > 22.0:
                        self.detector.cancel_threat()
                        is_violence = False
                        lbl = "NonViolence"
                        conf = min(conf, 22.0)

                elif has_victim and has_attacker and (conf >= 30.0 or is_violence or max_box_motion >= 2.0):
                    is_violence = True
                    lbl = "Violence"
                    conf = max(conf, 92.0)

                elif n_fighting >= 2 and (conf >= 10.0 or is_violence or max_box_motion >= 2.5):
                    is_violence = True
                    lbl = "Violence"
                    conf = max(conf, 90.0)

                elif (conf >= 70.0 or is_violence) and (
                    n_fighting >= 1 or (has_victim and has_attacker) or
                    (n_persons >= 2 and max_box_motion >= 2.2 and min_pair_dist <= 1.20)
                ):
                    is_violence = True
                    lbl = "Violence"
                    conf = max(conf, 88.0)

                has_physical_combat = (
                    (n_fighting >= 1) or
                    (has_victim and has_attacker) or
                    (n_persons >= 2 and max_box_motion >= 2.2 and min_pair_dist <= 1.20 and conf >= 70.0)
                )
                if processed_frames > 30 and is_violence and conf >= self.conf_threshold and has_physical_combat:
                    self._handle_alert(frame, lbl, conf)
                    if self.sound_enabled and (time.time() - self._last_beep > 1.2):
                        self._last_beep = time.time()
                        threading.Thread(target=trigger_audio_alarm, daemon=True).start()


                if out is not None:
                    out.write(frame)

                show = cv2.resize(frame, (dw, dh))

                self.hud.tick()
                self.hud.push_conf(conf)

                if self.ui_clean_mode:
                    cv2.putText(
                        show,
                        "[CLEAN FEED - Nhan TAB/C de hien HUD]",
                        (20, dh - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        (140, 160, 175),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    if self.ui_show_heatmap and self._ma:
                        hm = self._ma.get_flow_heatmap(show.shape)
                        if hm is not None:
                            cv2.addWeighted(show, 1.0, hm, self.heatmap_alpha, 0, show)

                    if self.ui_show_boxes and person_boxes:
                        show = draw_tactical_boxes(show, person_boxes, w, h, is_violence, conf)

                    show = self.hud.draw(
                        show, lbl, conf, is_violence, self.cam_label,
                        self.sound_enabled,
                        person_count=n_persons,
                        fighting_count=n_fighting,
                    )

                cv2.imshow(win, show)

                if is_stream:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    elapsed = time.time() - frame_start_t
                    remain_ms = int((frame_interval - elapsed) * 1000)
                    wait_ms = max(1, min(remain_ms, 33)) if remain_ms > 0 else 1
                    key = cv2.waitKey(wait_ms) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    log.info(f"[{self.cam_label}] Người dùng yêu cầu thoát.")
                    break
                elif key in (ord("h"), ord("H")):
                    self.ui_show_heatmap = not self.ui_show_heatmap
                    log.info(f"[{self.cam_label}] Heatmap: {'BẬT' if self.ui_show_heatmap else 'TẮT'}")
                elif key in (ord("b"), ord("B")):
                    self.ui_show_boxes = not self.ui_show_boxes
                    status_str = "BẬT (HIỆN KHUNG)" if self.ui_show_boxes else "TẮT (ẨN KHUNG)"
                    log.info(f"[{self.cam_label}] Khung Bounding Box: {status_str}")
                elif key in (9, ord("c"), ord("C")):
                    self.ui_clean_mode = not self.ui_clean_mode
                    status_str = "BẬT (ẨN TOÀN BỘ HUD & KHUNG)" if self.ui_clean_mode else "TẮT (HIỆN ĐẦY ĐỦ HUD)"
                    log.info(f"[{self.cam_label}] Chế độ Clean Feed: {status_str}")
                elif key in (ord("g"), ord("G")):
                    self.hud.show_graph = not self.hud.show_graph
                elif key in (ord("a"), ord("A")):
                    self.sound_enabled = not self.sound_enabled
                    log.info(f"[{self.cam_label}] Âm thanh cảnh báo: {'BẬT' if self.sound_enabled else 'TẮT'}")
                elif key in (ord("s"), ord("S")):
                    save_snapshot(frame, self.snap_folder, "MANUAL_SNAP", self.cam_label)
                elif key == ord(" "):
                    self.is_paused = not self.is_paused

        except KeyboardInterrupt:
            log.info("Dừng chương trình bởi Ctrl+C.")
        finally:
            grabber.stop()
            if out:
                out.release()
            try:
                cv2.destroyWindow(win)
            except Exception:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            dur = time.time() - self._session_start
            for notif in [self.discord, self.telegram]:
                if notif:
                    threading.Thread(
                        target=notif.send_session_summary,
                        args=(self.hud.alert_count, dur, self.cam_label),
                        daemon=True,
                    ).start()
            time.sleep(0.5)
            log.info(f"[{self.cam_label}] ══ Phiên kết thúc ══ Alerts: {self.hud.alert_count} | Uptime: {self.hud.uptime}")


def main():
    p = argparse.ArgumentParser(
        description="Violence Detection System ★ VIP PRO EDITION ★",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="Đường dẫn tệp cấu hình YAML")
    p.add_argument("--source", default=None, help="Nguồn video (File/Webcam/RTSP)")
    p.add_argument("--label", default=None, help="Tên hiển thị của camera")
    p.add_argument("--output", default=None, help="Tên tệp video xuất ra")
    p.add_argument("--no-save", action="store_true", help="Không ghi video ra đĩa")
    p.add_argument("--multi", action="store_true", help="Kích hoạt tất cả camera trong multi config")
    args = p.parse_args()

    cfg = load_config(args.config)
    m = cfg["model"]

    detector = ViolenceDetector(
        seq_model_path=m["seq_model_path"],
        input_size=m["input_size"],
        seq_len=m["sequence_length"],
    )

    lc = cfg.get("logging", {})
    det_logger = DetectionLogger(
        csv_path=lc.get("log_file", "detection_log.csv"),
        json_path=lc.get("json_log_file", "detection_log.json"),
    )

    discord, telegram = build_notifiers(cfg)
    save_output = not args.no_save

    if args.multi:
        cameras = cfg["source"].get("multi", [])
        if not cameras:
            log.error("Không có camera nào trong config.yaml > source > multi!")
            sys.exit(1)

        threads = []
        for i, cam in enumerate(cameras):
            label = cam.get("label", f"Cam{i + 1}")
            runner = CameraRunner(
                source=cam.get("url", ""),
                cfg=cfg,
                detector=detector,
                det_logger=det_logger,
                discord=discord,
                telegram=telegram,
                cam_label=label,
                save_output=save_output,
                out_filename=f"output_{label.replace(' ', '_')}.mp4",
            )
            t = threading.Thread(target=runner.run, name=f"cam-{label}", daemon=True)
            t.start()
            threads.append(t)
            log.info(f"[Multi] Đang kích hoạt luồng camera: {label}")

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            log.info("Dừng toàn bộ camera.")
        return

    source = args.source
    if source is None:
        source = cfg["source"]["default"]

    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cam_label = args.label or "Cam-Security"
    out_file = args.output or cfg["recording"]["output_filename"]

    CameraRunner(
        source=source,
        cfg=cfg,
        detector=detector,
        det_logger=det_logger,
        discord=discord,
        telegram=telegram,
        cam_label=cam_label,
        save_output=save_output,
        out_filename=out_file,
    ).run()


if __name__ == "__main__":
    main()
