"""集中管理所有可調整的參數。之後的規則門檻、視窗長度、判定參數也都放這裡。"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---------- 輸入 ----------
CAMERA_INDEX = 0            # 筆電內建鏡頭通常是 0，外接的試 1
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
# Windows 開鏡頭很慢或打不開時改成 True（改用 DirectShow）
USE_DSHOW_ON_WINDOWS = False

# ---------- 骨架模型 ----------
# 和 W2 用的同一個 .task 模型檔；FPS 太低可以換 pose_landmarker_lite.task
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_full.task"
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ---------- 特徵 ----------
# 關鍵點 visibility 低於此值就視為「沒拍到」，該幀不計算特徵
VISIBILITY_THRESHOLD = 0.5
# 腳踝專用門檻（筆電鏡頭俯角導致腳踝 confidence 偏低，單獨放寬）
ANKLE_VISIBILITY_THRESHOLD = 0.2
# 肩寬（像素）小於此值視為偵測異常，避免除以極小值
MIN_SHOULDER_WIDTH_PX = 20

# ---------- 顯示 ----------
MIRROR_DISPLAY = True       # 畫面左右翻轉像照鏡子；只影響顯示，不影響偵測
WINDOW_NAME = "Stretch Game - M1 viewer"

# ---------- 輸出 ----------
OUTPUT_DIR = BASE_DIR / "outputs"

# ---------- 校正（M2） ----------
CALIBRATION_SEC = 3          # 站立校正收集秒數
# 有效幀比例低於此值 → 校正失敗，要求重來
CALIBRATION_MIN_VALID_RATIO = 0.6
# 校正期間 hip_y_px 標準差 / 肩寬 超過此值 → 沒站穩，要求重來
CALIBRATION_MAX_HIP_STD_RATIO = 0.1

# ---------- 滑動視窗（M2） ----------
WINDOW_MS = 1000             # 視窗長度（毫秒）
WINDOW_STRIDE_MS = 200       # 每隔多久輸出一個視窗（毫秒）
MIN_VALID_RATIO = 0.7        # 視窗內有效幀比例低於此值 → 該視窗無效

# ---------- 規則分類器門檻（M2） ----------
# 跳躍
JUMP_RISE = 0.25             # hip_rise_max 超過此值才算跳起來
JUMP_RANGE = 0.35            # hip_rise_max - hip_rise_min 超過此值才算跳起落下
# 蹲下
SQUAT_DROP = -0.3            # hip_rise_mean 低於此值（髖部比站立低 0.3 個肩寬）
# 手臂判定
ARM_RAISED_MIN = 25          # 肩角 >= 此值才算舉起
ELBOW_STRAIGHT_MIN = 140     # 肘角 >= 此值才算手臂打直
ARM_STABLE_STD = 12          # 肩角 std <= 此值才算維持住（不是在移動中）
# 角度分界（45/90/135/180 四級）
ARM_ANGLE_BOUNDARIES = (67.5, 112.5, 145.0)

# ---------- classify 模式監控（M2） ----------
# 當前肩寬與校正基準相差超過此比例 → 提示「請回到原位」且不輸出預測
SHOULDER_WIDTH_DRIFT_RATIO = 0.25

# ---------- 字型（classify 模式中文 HUD；M4 遊戲畫面沿用） ----------
# Windows：微軟正黑體；macOS：蘋方；找不到時 put_text_pil 退回英文
FONT_PATH_WIN = "C:/Windows/Fonts/msjh.ttc"
FONT_PATH_MAC = "/System/Library/Fonts/PingFang.ttc"
import platform as _platform
FONT_PATH = FONT_PATH_WIN if _platform.system() == "Windows" else FONT_PATH_MAC

# ---------- 判定機制（M3） ----------
# 連續 N 個有效視窗相同預測 → 判定成立
CONFIRM_WINDOWS = 3
# 機率需超過此門檻才計入連續計數（規則法固定 1.0；ML 分類器用）
PROB_THRESHOLD = 0.7
# 判定成立後在畫面顯示大字提示的持續時間（毫秒）
CONFIRM_DISPLAY_MS = 1000

# ---------- 分類器切換（預留給 M6 ML 分類器） ----------
CLASSIFIER = "rule"          # "rule" | "ml"
