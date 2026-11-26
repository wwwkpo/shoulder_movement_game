"""Stretch Game 主入口。

用法：
    python main.py                       # 預設 viewer 模式（M1 除錯畫面）
    python main.py --mode viewer         # M1 除錯畫面（骨架、角度、逐幀特徵）
    python main.py --mode classify       # M2 校正 + 即時分類（規則法）
    python main.py --video assets/xx.mp4 # 用影片檔（可重複測試）
    python main.py --mode classify --log # 輸出視窗特徵 CSV 到 outputs/

按鍵（所有模式）：
    q / Esc  離開
    m        切換鏡像顯示
    空白鍵   暫停 / 繼續

classify 模式額外按鍵：
    c        隨時重新校正
"""
import argparse
import csv
import time
from datetime import datetime

import cv2

import config
import ui
from calibration import Calibrator
from classifiers.rule import RuleClassifier
from features import FEATURE_NAMES, compute_frame_features
from pose import PoseEstimator
from source import FrameSource
from window import SlidingWindow, WINDOW_FEATURE_NAMES


# ------------------------------------------------------------------ #
# 共用工具
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description="Stretch game")
    parser.add_argument(
        "--mode", choices=["viewer", "classify"], default="viewer",
        help="viewer=M1 除錯畫面；classify=M2 校正+即時分類（預設 viewer）",
    )
    parser.add_argument("--video", help="影片檔路徑；不給就用 Webcam")
    parser.add_argument("--log", action="store_true", help="輸出特徵 CSV 到 outputs/")
    return parser.parse_args()


# ------------------------------------------------------------------ #
# M1 Viewer 模式
# ------------------------------------------------------------------ #

def _open_frame_logger():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"frame_features_{datetime.now():%Y%m%d_%H%M%S}.csv"
    stream = path.open("w", encoding="utf-8-sig", newline="")
    columns = ["timestamp_ms", "valid", *FEATURE_NAMES, "hip_y_px", "shoulder_width_px"]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    return path, stream, writer


def run_viewer(args):
    source = FrameSource(args.video)
    estimator = PoseEstimator()
    mirrored = config.MIRROR_DISPLAY
    paused = False

    log_path = log_stream = log_writer = None
    if args.log:
        log_path, log_stream, log_writer = _open_frame_logger()

    fps_smoothed = 0.0
    last_time = time.monotonic()
    display = None

    try:
        while True:
            if not paused:
                frame, ts = source.read()
                if frame is None:
                    print("影片結束或讀不到畫面。")
                    break

                landmarks = estimator.detect(frame, ts)
                features = raw = None
                if landmarks is None:
                    status, ok = "No person detected", False
                else:
                    features, raw, missing = compute_frame_features(landmarks)
                    ui.draw_skeleton(frame, landmarks, config.VISIBILITY_THRESHOLD)
                    if missing:
                        status, ok = "Step back / full body: " + ", ".join(missing[:4]), False
                    else:
                        status, ok = "OK", True

                display = cv2.flip(frame, 1) if mirrored else frame
                if features is not None:
                    ui.draw_angle_labels(display, landmarks, features, mirrored)

                now = time.monotonic()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 / dt if fps_smoothed else 1.0 / dt
                ui.draw_hud(display, fps_smoothed, status, ok, features, raw)

                if log_writer is not None:
                    row = {"timestamp_ms": ts, "valid": int(features is not None)}
                    if features is not None:
                        row.update(features)
                        row.update(raw)
                    log_writer.writerow(row)

            cv2.imshow(config.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirrored = not mirrored
            if key == ord(" "):
                paused = not paused
    finally:
        source.release()
        estimator.close()
        cv2.destroyAllWindows()
        if log_stream is not None:
            log_stream.close()
            print(f"逐幀特徵已存到：{log_path}")


# ------------------------------------------------------------------ #
# M2 Classify 模式
# ------------------------------------------------------------------ #

def _open_window_logger():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / f"window_features_{datetime.now():%Y%m%d_%H%M%S}.csv"
    stream = path.open("w", encoding="utf-8-sig", newline="")
    columns = ["timestamp_ms", "pred", "prob", *WINDOW_FEATURE_NAMES]
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    return path, stream, writer


def _draw_calibrating(display, cal_msg: str, pose_status: str, fps: float):
    """校正期間 overlay：中文大字提示 + FPS + 姿態狀態。"""
    h, w = display.shape[:2]
    # 中文校正訊息（大字，螢幕垂直中央）
    ui.put_text_pil(display, cal_msg, (w // 2 - 220, h // 2 - 20),
                    color=ui.YELLOW, font_size=36)
    ui.put_text(display, f"FPS {fps:4.1f}", (10, 25))
    ok = pose_status == "OK"
    ui.put_text(display, pose_status, (10, 52), ui.GREEN if ok else ui.RED, 0.65, 2)


def _draw_classify_hud(display, fps: float, pose_status: str,
                       prediction: str, explain_lines: list[str],
                       out_of_position: bool):
    """分類模式 HUD：FPS、姿態狀態、預測結果、explain 明細、離位警告。"""
    h, w = display.shape[:2]

    # FPS 與姿態狀態
    ok = pose_status == "OK"
    ui.put_text(display, f"FPS {fps:4.1f}", (10, 25))
    ui.put_text(display, pose_status, (10, 52), ui.GREEN if ok else ui.RED, 0.65, 2)

    # 預測 label（中文大字，螢幕下方中央）
    ui.put_text_pil(display, f"預測：{prediction}", (w // 2 - 100, h - 65),
                    color=(128, 255, 0), font_size=36)

    # explain 明細（右側，小字 ASCII；✓✗替換為 cv2 相容字元）
    x_explain = max(w - 430, w // 2 + 10)
    y = 30
    for line in explain_lines:
        line_ascii = line.replace("✓", "[ok]").replace("✗", "[--]")
        color = ui.GREEN if "[ok]" in line_ascii else ui.RED if "[--]" in line_ascii else ui.WHITE
        ui.put_text(display, line_ascii, (x_explain, y), color, 0.42, 1)
        y += 17

    # 離位警告（中文，螢幕中央，紅色）
    if out_of_position:
        ui.put_text_pil(display, "請回到原位或按 c 重新校正",
                        (w // 2 - 230, h // 2 - 25),
                        color=ui.RED, font_size=34)


def run_classify(args):
    source = FrameSource(args.video)
    estimator = PoseEstimator()
    classifier = RuleClassifier()
    win = SlidingWindow()
    mirrored = config.MIRROR_DISPLAY
    paused = False

    log_path = log_stream = log_writer = None
    if args.log:
        log_path, log_stream, log_writer = _open_window_logger()

    # 狀態機：calibrating → classifying
    calibrating = True
    calibrator = Calibrator()
    calib = None
    cal_msg = "請自然站立，校正中…"

    prediction = "---"
    prob = 1.0
    explain_lines: list[str] = []
    out_of_position = False

    fps_smoothed = 0.0
    last_time = time.monotonic()
    display = None

    print("校正模式：請自然站立，等待 3 秒後開始分類。")
    print("按 c 可隨時重新校正，q 離開。")

    try:
        while True:
            if not paused:
                frame, ts = source.read()
                if frame is None:
                    print("影片結束或讀不到畫面。")
                    break

                # 偵測姿態（一律對未翻轉的原始影格）
                landmarks = estimator.detect(frame, ts)
                features = raw = None
                pose_status = "No person detected"
                if landmarks is not None:
                    features, raw, missing = compute_frame_features(landmarks)
                    ui.draw_skeleton(frame, landmarks, config.VISIBILITY_THRESHOLD)
                    if missing:
                        pose_status = "Step back / full body"
                    else:
                        pose_status = "OK"

                # 顯示時才鏡像（骨架已畫在原始影格上）
                display = cv2.flip(frame, 1) if mirrored else frame
                if features is not None:
                    ui.draw_angle_labels(display, landmarks, features, mirrored)

                # FPS
                now = time.monotonic()
                dt = now - last_time
                last_time = now
                if dt > 0:
                    fps_smoothed = 0.9 * fps_smoothed + 0.1 / dt if fps_smoothed else 1.0 / dt

                # ---- 校正模式 ----
                if calibrating:
                    cal_msg, done = calibrator.feed(ts, features, raw)
                    if done:
                        if calibrator.success:
                            calib = calibrator.result
                            calibrating = False
                            prediction = "---"
                            explain_lines = []
                            print("校正成功，開始分類。")
                        else:
                            print(f"[校正失敗] {cal_msg}  → 自動重試")
                            calibrator = Calibrator()
                    _draw_calibrating(display, cal_msg, pose_status, fps_smoothed)

                # ---- 分類模式 ----
                else:
                    if features is not None and raw is not None:
                        if calib.shoulder_width_ok(raw):
                            # 在原位：加入 hip_rise，餵入視窗
                            out_of_position = False
                            feat_ext = calib.add_hip_rise(features, raw)
                            win_result = win.feed(ts, feat_ext)
                        else:
                            # 離開原位：無效幀餵入視窗，不輸出預測
                            out_of_position = True
                            win_result = win.feed(ts, None)
                    else:
                        # 偵測失敗：無效幀
                        out_of_position = False
                        win_result = win.feed(ts, None)

                    if win_result is not None and not out_of_position:
                        prediction, prob = classifier.predict(win_result)
                        explain_lines = classifier.explain(win_result)
                        if log_writer is not None:
                            row = {"timestamp_ms": ts, "pred": prediction, "prob": prob}
                            row.update(win_result)
                            log_writer.writerow(row)

                    _draw_classify_hud(display, fps_smoothed, pose_status,
                                       prediction, explain_lines, out_of_position)

            cv2.imshow(config.WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirrored = not mirrored
            if key == ord(" "):
                paused = not paused
            if key == ord("c") and not calibrating:
                # 隨時按 c 重新校正
                calibrating = True
                calibrator = Calibrator()
                win.reset()
                prediction = "---"
                explain_lines = []
                out_of_position = False
                cal_msg = "請自然站立，校正中…"
                print("重新校正中…")
    finally:
        source.release()
        estimator.close()
        cv2.destroyAllWindows()
        if log_stream is not None:
            log_stream.close()
            print(f"視窗特徵已存到：{log_path}")


# ------------------------------------------------------------------ #

def main():
    args = parse_args()
    if args.mode == "classify":
        run_classify(args)
    else:
        run_viewer(args)


if __name__ == "__main__":
    main()
