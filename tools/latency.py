"""工具：從 --log 輸出的 CSV 計算判定延遲。

用法：
    python tools/latency.py \\
        --frames  outputs/frame_features_YYYYmmdd_HHMMSS.csv \\
        --windows outputs/window_features_YYYYmmdd_HHMMSS.csv \\
        [--window-ms 1000]

延遲定義（逐幀起點）：
  側舉  → 肩角第一次進入目標角度區間且之後持續維持到判定的那一幀
  蹲下  → hip_rise 第一次低於 SQUAT_DROP 且之後持續維持的那一幀
  跳躍  → hip_rise 第一次超過 JUMP_RISE  且之後持續維持的那一幀

延遲 = 判定成立 timestamp − 姿勢到位 timestamp

輸出各類別的平均與最大延遲（毫秒），以及全部事件的彙總。
"""
import argparse
import csv
import sys
from pathlib import Path

# 允許直接執行（project root 不在 sys.path 時自動加入）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


# ------------------------------------------------------------------ #
# 讀取 CSV

def _read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# ------------------------------------------------------------------ #
# 姿勢到位判斷：找「最後一段連續滿足條件的起始幀」

def _last_continuous_entry(frames: list[dict], col: str, lo: float, hi: float) -> int | None:
    """
    在 frames（已按 timestamp_ms 升序排列）中，找
    「最後一段連續滿足 lo <= col <= hi 的 streak 的起始 timestamp_ms」。
    frames 中無效幀（valid=0 或 col 為空）被視為條件不滿足（打斷 streak）。

    回傳 int timestamp_ms，或 None（找不到任何滿足的幀）。
    """
    streak_start: int | None = None
    last_streak_start: int | None = None

    for row in frames:
        if row.get("valid", "0") != "1":
            streak_start = None
            continue
        raw_val = row.get(col, "")
        if raw_val == "" or raw_val is None:
            streak_start = None
            continue
        try:
            val = float(raw_val)
        except ValueError:
            streak_start = None
            continue

        if lo <= val <= hi:
            if streak_start is None:
                streak_start = int(row["timestamp_ms"])
        else:
            if streak_start is not None:
                # 這個 streak 中途斷了，更新 last_streak_start
                last_streak_start = streak_start
            streak_start = None

    # 迴圈結束時若 streak 仍在進行（即一直到最後都滿足）
    if streak_start is not None:
        last_streak_start = streak_start

    return last_streak_start


def _pose_ready_ts(frames: list[dict], label: str, confirm_ts: int) -> int | None:
    """
    在 timestamp_ms <= confirm_ts 的幀中，找姿勢到位的時間戳記。
    回傳 int 或 None。
    """
    eligible = [r for r in frames if int(r["timestamp_ms"]) <= confirm_ts]
    if not eligible:
        return None

    if label == "jump":
        # hip_rise 第一次超過 JUMP_RISE 且之後持續維持
        return _last_continuous_entry(eligible, "hip_rise", config.JUMP_RISE, 1e9)

    if label == "squat":
        # hip_rise 第一次低於 SQUAT_DROP 且之後持續維持
        return _last_continuous_entry(eligible, "hip_rise", -1e9, config.SQUAT_DROP)

    # 側舉：left_N 或 right_N
    if label.startswith("left_") or label.startswith("right_"):
        try:
            grade = int(label.split("_")[1])
        except (IndexError, ValueError):
            return None
        side = "L" if label.startswith("left_") else "R"
        col = f"{side}_shoulder_angle"

        bounds = config.ARM_ANGLE_BOUNDARIES  # (67.5, 112.5, 145.0)
        grade_ranges = {
            45:  (config.ARM_RAISED_MIN, bounds[0]),
            90:  (bounds[0], bounds[1]),
            135: (bounds[1], bounds[2]),
            180: (bounds[2], 360.0),
        }
        lo, hi = grade_ranges.get(grade, (0.0, 360.0))
        return _last_continuous_entry(eligible, col, lo, hi)

    return None


# ------------------------------------------------------------------ #
# 主要分析函式

def analyze(frames_path: str, windows_path: str, window_ms: int | None = None) -> None:
    frames = _read_csv(frames_path)
    windows = _read_csv(windows_path)

    # 只看 confirmed=1 的視窗
    confirmed_rows = [r for r in windows if r.get("confirmed", "0") == "1"]
    if not confirmed_rows:
        print("找不到任何 confirmed=1 的視窗，請確認 CSV 是否來自 --log 模式。")
        return

    label_header = f"視窗大小：{window_ms}ms" if window_ms else "（未指定視窗大小）"
    print(f"\n{label_header}")
    print(f"frames CSV  : {frames_path}")
    print(f"windows CSV : {windows_path}")
    print(f"confirmed 事件數：{len(confirmed_rows)}\n")

    # 收集每個事件的延遲
    per_class: dict[str, list[int]] = {}
    skipped = 0

    for row in confirmed_rows:
        label = row.get("pred", "")
        if label in ("none", ""):
            continue
        try:
            confirm_ts = int(row["timestamp_ms"])
        except (KeyError, ValueError):
            skipped += 1
            continue

        pose_ts = _pose_ready_ts(frames, label, confirm_ts)
        if pose_ts is None:
            # 找不到姿勢到位的幀（可能發生在校正剛結束時）
            skipped += 1
            continue

        latency = confirm_ts - pose_ts
        if latency < 0:
            # 資料異常（confirm 早於 pose_ready，不可能發生）
            skipped += 1
            continue

        per_class.setdefault(label, []).append(latency)

    if skipped:
        print(f"  （略過 {skipped} 個無法計算延遲的事件）\n")

    # 輸出表格
    all_latencies: list[int] = []
    print(f"{'類別':<18}  {'平均延遲(ms)':>12}  {'最大延遲(ms)':>12}  {'次數':>4}")
    print("-" * 55)
    for lbl in sorted(per_class):
        vals = per_class[lbl]
        avg = sum(vals) / len(vals)
        mx  = max(vals)
        all_latencies.extend(vals)
        print(f"{lbl:<18}  {avg:>12.0f}  {mx:>12}  {len(vals):>4}")

    if all_latencies:
        print("-" * 55)
        print(f"{'全部':<18}  {sum(all_latencies)/len(all_latencies):>12.0f}"
              f"  {max(all_latencies):>12}  {len(all_latencies):>4}")
    print()


# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="計算判定延遲（M3 工具）")
    parser.add_argument("--frames",  required=True, help="逐幀特徵 CSV 路徑")
    parser.add_argument("--windows", required=True, help="視窗特徵 CSV 路徑")
    parser.add_argument("--window-ms", type=int, default=None,
                        help="視窗大小（毫秒，僅用於報告標題）")
    args = parser.parse_args()

    analyze(args.frames, args.windows, args.window_ms)


if __name__ == "__main__":
    main()
