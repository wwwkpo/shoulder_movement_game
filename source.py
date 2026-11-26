"""影像輸入：Webcam 或影片檔，統一回傳 (frame_bgr, timestamp_ms)。

時間戳記一律用「真實時間」(webcam) 或「影格序號 / fps」(影片)，
後面的滑動視窗會用時間戳記計算長度，不寫死幀數，避免實際 FPS 掉下來時視窗長度跑掉。
"""
import math
import platform
import time
from pathlib import Path

import cv2

import config


class FrameSource:
    def __init__(self, video_path=None):
        self.is_video = video_path is not None
        if self.is_video:
            video_path = Path(video_path)
            if not video_path.is_file():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            self.cap = cv2.VideoCapture(str(video_path))
            fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        else:
            if config.USE_DSHOW_ON_WINDOWS and platform.system() == "Windows":
                self.cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            self.fps = None
        if not self.cap.isOpened():
            target = video_path if self.is_video else f"camera {config.CAMERA_INDEX}"
            raise RuntimeError(f"Cannot open {target}")
        self.frame_index = 0
        self._t0 = time.monotonic()
        self._last_ts = -1

    @property
    def size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def read(self):
        """回傳 (frame_bgr, timestamp_ms)；讀不到時回傳 (None, None)。"""
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        if self.is_video:
            ts = int(self.frame_index / self.fps * 1000)
        else:
            ts = int((time.monotonic() - self._t0) * 1000)
        # detect_for_video 要求時間戳記嚴格遞增
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        self.frame_index += 1
        return frame, ts

    def release(self):
        self.cap.release()
