"""骨架偵測：沿用 W2 的 MediaPipe Tasks API（PoseLandmarker + detect_for_video）。

回傳每幀一個 (33, 4) 的 numpy 陣列：x_px, y_px, z, visibility。
x、y 已換成「像素座標」——MediaPipe 的 x 是除以寬、y 是除以高，
16:9 畫面直接用正規化座標算夾角會失真，所以一律先乘回像素。
"""
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config

# MediaPipe Pose 33 點中本專案用到的索引（左右是「受試者本人」的左右）
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

# 畫骨架用的連線（只畫身體主幹，不畫臉部細節）
SKELETON_CONNECTIONS = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


class PoseEstimator:
    def __init__(self, model_path=None):
        model_path = model_path or config.MODEL_PATH
        if not model_path.is_file():
            raise FileNotFoundError(
                f"找不到模型檔：{model_path}\n"
                "請把 W2 用的 pose_landmarker_full.task（或 lite）放到 models/ 資料夾。"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms):
        """回傳 (33, 4) 陣列 [x_px, y_px, z, visibility]；沒偵測到人時回傳 None。"""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        lms = result.pose_landmarks[0]
        return np.array(
            [[lm.x * w, lm.y * h, lm.z, lm.visibility] for lm in lms],
            dtype=np.float64,
        )

    def close(self):
        self.landmarker.close()
