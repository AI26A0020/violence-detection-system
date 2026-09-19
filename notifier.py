import io
import time
import logging
import requests
import cv2

logger = logging.getLogger(__name__)


class DiscordNotifier:

    def __init__(self, webhook_url: str, mention_role_id: str = ""):
        self.webhook_url = webhook_url
        self.mention = f"<@&{mention_role_id}> " if mention_role_id else ""

    def send_alert(self, label: str, confidence: float, frame=None, camera_label: str = "Camera"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        content = (
            f"{self.mention}🚨 **CẢNH BÁO BẠO LỰC PHÁT HIỆN!**\n"
            f"📷 Camera: **{camera_label}**\n"
            f"🎯 Nhãn: **{label}**\n"
            f"📊 Độ tin cậy: **{confidence:.1f}%**\n"
            f"🕐 Thời gian: `{timestamp}`"
        )

        files = {}
        if frame is not None:
            try:
                success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    files["file"] = ("alert.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")
            except Exception as e:
                logger.warning(f"[Discord] Không encode được ảnh: {e}")

        try:
            if files:
                resp = requests.post(
                    self.webhook_url,
                    data={"content": content},
                    files=files,
                    timeout=10,
                )
            else:
                resp = requests.post(
                    self.webhook_url,
                    json={"content": content},
                    timeout=10,
                )
            resp.raise_for_status()
            logger.info(f"[Discord] Gửi alert thành công ({resp.status_code})")
        except requests.RequestException as e:
            logger.error(f"[Discord] Gửi thất bại: {e}")

    def send_session_summary(self, total_alerts: int, duration_seconds: float, camera_label: str = "Camera"):
        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)
        content = (
            f"📋 **Kết thúc phiên giám sát** — {camera_label}\n"
            f"⏱️ Thời lượng: `{mins}m {secs}s`\n"
            f"🚨 Tổng cảnh báo: **{total_alerts}** lần"
        )
        try:
            requests.post(self.webhook_url, json={"content": content}, timeout=10)
        except requests.RequestException as e:
            logger.error(f"[Discord] Gửi summary thất bại: {e}")


class TelegramNotifier:

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

    def send_alert(self, label: str, confidence: float, frame=None, camera_label: str = "Camera"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"🚨 *CẢNH BÁO BẠO LỰC*\n"
            f"📷 Camera: *{camera_label}*\n"
            f"🎯 Nhãn: *{label}*\n"
            f"📊 Độ tin cậy: *{confidence:.1f}%*\n"
            f"🕐 Thời gian: `{timestamp}`"
        )

        if frame is not None:
            try:
                success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    resp = requests.post(
                        f"{self.base_url}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": text, "parse_mode": "Markdown"},
                        files={"photo": ("alert.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
                        timeout=15,
                    )
                    resp.raise_for_status()
                    logger.info(f"[Telegram] Gửi ảnh alert thành công")
                    return
            except Exception as e:
                logger.warning(f"[Telegram] Không gửi được ảnh: {e}")

        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
            resp.raise_for_status()
            logger.info("[Telegram] Gửi text alert thành công")
        except requests.RequestException as e:
            logger.error(f"[Telegram] Gửi thất bại: {e}")

    def send_session_summary(self, total_alerts: int, duration_seconds: float, camera_label: str = "Camera"):
        mins = int(duration_seconds // 60)
        secs = int(duration_seconds % 60)
        text = (
            f"📋 *Kết thúc phiên giám sát* — {camera_label}\n"
            f"⏱️ Thời lượng: `{mins}m {secs}s`\n"
            f"🚨 Tổng cảnh báo: *{total_alerts}* lần"
        )
        try:
            requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
                timeout=10,
            )
        except requests.RequestException as e:
            logger.error(f"[Telegram] Gửi summary thất bại: {e}")
