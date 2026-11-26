"""逐幀特徵計算：正規化、三點夾角、相對位置。

規則法和之後的機器學習模型都用這裡算出來的同一份特徵。

座標約定（正規化後）：
- 原點：髖部中心
- 單位：肩寬（像素）
- y 軸：往上為正（影像座標是往下為正，這裡翻過來比較直覺）
- x 軸：往受試者本人的「左邊」為正；用左右肩的相對位置決定方向，
        所以畫面有沒有鏡像翻轉都不影響特徵。
"""
import math

import numpy as np

import config
from pose import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)

REQUIRED_POINTS = {
    "L_shoulder": L_SHOULDER, "R_shoulder": R_SHOULDER,
    "L_elbow": L_ELBOW, "R_elbow": R_ELBOW,
    "L_wrist": L_WRIST, "R_wrist": R_WRIST,
    "L_hip": L_HIP, "R_hip": R_HIP,
    "L_knee": L_KNEE, "R_knee": R_KNEE,
    "L_ankle": L_ANKLE, "R_ankle": R_ANKLE,
}

# 特徵欄位順序固定，CSV 與之後的模型都依這個順序
FEATURE_NAMES = [
    "L_shoulder_angle", "R_shoulder_angle",
    "L_elbow_angle", "R_elbow_angle",
    "L_knee_angle", "R_knee_angle",
    "L_wrist_dx", "L_wrist_dy",      # 手腕相對同側肩膀；dx 往身體外側為正
    "R_wrist_dx", "R_wrist_dy",
    "hip_ankle_height",               # 髖部中心到腳踝中心的高度（蹲下會變小）
    "torso_tilt",                     # 軀幹相對垂直線的側傾角度，往本人左側為正
]


def calculate_angle(a, vertex, c):
    """三點夾角（度），以 vertex 為頂點。沿用 W4 的算法，改成通用版本。"""
    v1 = a - vertex
    v2 = c - vertex
    length_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if length_product <= 1e-12:
        return None
    cosine = np.dot(v1, v2) / length_product
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def check_visibility(landmarks, threshold=None):
    """回傳 visibility 不足的關鍵點名稱清單；空清單代表全部看得到。"""
    threshold = config.VISIBILITY_THRESHOLD if threshold is None else threshold
    return [name for name, idx in REQUIRED_POINTS.items()
            if landmarks[idx, 3] < threshold]


def compute_frame_features(landmarks):
    """
    輸入：pose.PoseEstimator.detect() 的 (33, 4) 陣列（像素座標）
    輸出：(features, raw, missing)
      features: dict，鍵為 FEATURE_NAMES；資料不足時為 None
      raw:      之後校正/判定會用到的原始量（髖部像素高度、肩寬像素）
      missing:  看不到的關鍵點清單
    """
    missing = check_visibility(landmarks)
    if missing:
        return None, None, missing

    p = landmarks[:, :2]  # 只用 2D 像素座標
    shoulder_w = float(np.linalg.norm(p[L_SHOULDER] - p[R_SHOULDER]))
    if shoulder_w < config.MIN_SHOULDER_WIDTH_PX:
        return None, None, ["shoulder_width_too_small"]

    hip_c = (p[L_HIP] + p[R_HIP]) / 2
    shoulder_c = (p[L_SHOULDER] + p[R_SHOULDER]) / 2
    ankle_c = (p[L_ANKLE] + p[R_ANKLE]) / 2

    # 受試者「左邊」在影像中的 x 方向（+1 或 -1）
    left_dir = 1.0 if p[L_SHOULDER][0] >= p[R_SHOULDER][0] else -1.0

    def norm(pt):
        """像素座標 -> 正規化座標（原點髖部中心、單位肩寬、y 往上、x 往本人左邊）。"""
        return np.array([(pt[0] - hip_c[0]) * left_dir / shoulder_w,
                         (hip_c[1] - pt[1]) / shoulder_w])

    def angle(i, j, k):
        return calculate_angle(p[i], p[j], p[k])

    f = {}
    # 肩角 = 上臂（肩→肘）與軀幹（肩→髖）的夾角
    f["L_shoulder_angle"] = angle(L_ELBOW, L_SHOULDER, L_HIP)
    f["R_shoulder_angle"] = angle(R_ELBOW, R_SHOULDER, R_HIP)
    f["L_elbow_angle"] = angle(L_SHOULDER, L_ELBOW, L_WRIST)
    f["R_elbow_angle"] = angle(R_SHOULDER, R_ELBOW, R_WRIST)
    f["L_knee_angle"] = angle(L_HIP, L_KNEE, L_ANKLE)
    f["R_knee_angle"] = angle(R_HIP, R_KNEE, R_ANKLE)

    # 手腕相對同側肩膀；左手往外 = 正規化 x 的 +，右手往外 = 正規化 x 的 -
    l_rel = norm(p[L_WRIST]) - norm(p[L_SHOULDER])
    r_rel = norm(p[R_WRIST]) - norm(p[R_SHOULDER])
    f["L_wrist_dx"], f["L_wrist_dy"] = float(l_rel[0]), float(l_rel[1])
    f["R_wrist_dx"], f["R_wrist_dy"] = float(-r_rel[0]), float(r_rel[1])

    f["hip_ankle_height"] = float((ankle_c[1] - hip_c[1]) / shoulder_w)

    torso = norm(shoulder_c)  # 髖部中心 -> 肩膀中心（髖部中心是原點）
    f["torso_tilt"] = float(math.degrees(math.atan2(torso[0], torso[1])))

    if any(v is None for v in f.values()):
        return None, None, ["degenerate_angle"]

    raw = {
        "hip_y_px": float(hip_c[1]),        # 跳躍/蹲下要跟「站立校正基準」比
        "shoulder_width_px": shoulder_w,
    }
    return f, raw, []
