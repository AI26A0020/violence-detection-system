import csv
import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class DetectionLogger:

    def __init__(self, csv_path: str = "detection_log.csv", json_path: str = "detection_log.json"):
        self.csv_path = csv_path
        self.json_path = json_path
        self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "camera", "label", "confidence", "snapshot_path"])
            logger.info(f"[Logger] Khởi tạo CSV log: {self.csv_path}")

    def log_event(
        self,
        label: str,
        confidence: float,
        camera_label: str = "Camera",
        snapshot_path: str = "",
    ):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, camera_label, label, f"{confidence:.2f}", snapshot_path])
        except Exception as e:
            logger.error(f"[Logger] Lỗi ghi CSV: {e}")

        event = {
            "timestamp": timestamp,
            "camera": camera_label,
            "label": label,
            "confidence": round(confidence, 2),
            "snapshot": snapshot_path,
        }
        try:
            events = []
            if os.path.exists(self.json_path):
                with open(self.json_path, "r", encoding="utf-8") as f:
                    try:
                        events = json.load(f)
                    except json.JSONDecodeError:
                        events = []
            events.append(event)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(events, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[Logger] Lỗi ghi JSON: {e}")

        logger.info(f"[Logger] Đã ghi event: {label} ({confidence:.1f}%) — {camera_label}")

    def get_today_stats(self, camera_label: str = None) -> dict:
        today = time.strftime("%Y-%m-%d")
        violence_count = 0
        total_count = 0
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row["timestamp"].startswith(today):
                        continue
                    if camera_label and row["camera"] != camera_label:
                        continue
                    total_count += 1
                    if row["label"] == "Violence":
                        violence_count += 1
        except Exception:
            pass
        return {"date": today, "total": total_count, "violence": violence_count}
