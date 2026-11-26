"""畫面繪製。

順序很重要：
1. 骨架等「貼在身體上」的圖形畫在原始影格上
2. 需要鏡像時整張翻轉
3. 文字（HUD）最後才畫，否則文字也會被左右翻轉

cv2.putText 不支援中文，除錯畫面先用英文；
中文訊息（校正提示、離位警告）用 put_text_pil（PIL）繪製。
"""
import numpy as np
import cv2

from pose import (
    L_ELBOW, L_KNEE, L_SHOULDER, R_ELBOW, R_KNEE, R_SHOULDER,
    SKELETON_CONNECTIONS,
)

GREEN = (0, 220, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# 在關節旁標示角度：特徵名稱 -> 關節索引
ANGLE_LABELS = {
    "L_shoulder_angle": L_SHOULDER, "R_shoulder_angle": R_SHOULDER,
    "L_elbow_angle": L_ELBOW, "R_elbow_angle": R_ELBOW,
    "L_knee_angle": L_KNEE, "R_knee_angle": R_KNEE,
}


def draw_skeleton(frame, landmarks, visibility_threshold):
    for i, j in SKELETON_CONNECTIONS:
        if landmarks[i, 3] < visibility_threshold or landmarks[j, 3] < visibility_threshold:
            continue
        p1 = tuple(int(v) for v in landmarks[i, :2])
        p2 = tuple(int(v) for v in landmarks[j, :2])
        cv2.line(frame, p1, p2, GREEN, 3, cv2.LINE_AA)
    for idx in {k for pair in SKELETON_CONNECTIONS for k in pair}:
        color = GREEN if landmarks[idx, 3] >= visibility_threshold else RED
        cv2.circle(frame, tuple(int(v) for v in landmarks[idx, :2]), 5, color, -1, cv2.LINE_AA)


def put_text(frame, text, org, color=WHITE, scale=0.6, thickness=1):
    """文字墊深色底，任何背景都看得清楚。"""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 3, y - th - 4), (x + tw + 3, y + baseline + 2), BLACK, -1)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_angle_labels(frame, landmarks, features, mirrored):
    """在關節旁標角度（要在翻轉之後呼叫，所以自己換算鏡像座標）。"""
    w = frame.shape[1]
    for name, idx in ANGLE_LABELS.items():
        x, y = landmarks[idx, :2]
        if mirrored:
            x = w - x
        put_text(frame, f"{features[name]:.0f}", (int(x) + 8, int(y) - 8), YELLOW, 0.6, 2)


def put_text_pil(frame, text: str, org: tuple[int, int],
                 color: tuple[int, int, int] = WHITE, font_size: int = 28):
    """用 PIL 在 frame 上繪製中文文字（帶黑底以提高可讀性）。
    找不到字型或 PIL 不可用時退回 put_text（僅顯示 ASCII）。
    org 為左上角座標 (x, y)，color 為 BGR。
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import config
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype(config.FONT_PATH, font_size)
        except (IOError, OSError):
            font = ImageFont.load_default()
        x, y = org
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle([bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3], fill=(0, 0, 0))
        r, g, b = color[2], color[1], color[0]   # BGR → RGB
        draw.text((x, y), text, font=font, fill=(r, g, b))
        frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        ascii_text = text.encode("ascii", "ignore").decode("ascii")
        put_text(frame, ascii_text, org, color)


def draw_hud(frame, fps, status, status_ok, features, raw):
    put_text(frame, f"FPS {fps:4.1f}", (10, 25))
    put_text(frame, status, (10, 52), GREEN if status_ok else RED, 0.65, 2)
    if features is None:
        return
    rows = [(n, f"{v:.2f}", WHITE) for n, v in features.items()]
    if raw:
        rows += [(n, f"{v:.1f}", (180, 180, 180)) for n, v in raw.items()]
    # 字型不是等寬，名稱和數值分兩欄對齊
    y = 85
    cv2.rectangle(frame, (5, y - 16), (265, y + 20 * (len(rows) - 1) + 6), BLACK, -1)
    for name, value, color in rows:
        cv2.putText(frame, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (190, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 20
