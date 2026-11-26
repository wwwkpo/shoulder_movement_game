"""驗證滑動視窗的切法、統計值、無效視窗處理。

執行：python tests/test_window.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from window import SlidingWindow, WINDOW_FEATURE_NAMES


# ------------------------------------------------------------------ #
# 測試輔助

def make_features(val: float) -> dict:
    """產生一個所有 13 個特徵都等於 val 的 features dict。"""
    from features import FEATURE_NAMES
    keys = FEATURE_NAMES + ["hip_rise"]
    return {k: val for k in keys}


def feed_sequence(win: SlidingWindow, timestamps, values):
    """
    批次餵入，回傳所有非 None 的視窗結果清單。
    values 為 float 清單（會轉成 make_features）或 None（無效幀）。
    """
    results = []
    for ts, v in zip(timestamps, values):
        feat = make_features(v) if v is not None else None
        r = win.feed(ts, feat)
        if r is not None:
            results.append(r)
    return results


# ------------------------------------------------------------------ #
# 測試 1：第一個視窗要等到緩衝涵蓋完整 WINDOW_MS 才輸出

def test_first_window_waits_full_duration():
    win = SlidingWindow()
    # 以 10ms 間隔送幀，WINDOW_MS 預設 1000ms
    # 在未達到 1000ms 前不應有輸出
    partial_results = []
    for t in range(0, config.WINDOW_MS, 50):
        r = win.feed(t, make_features(1.0))
        if r is not None:
            partial_results.append(r)
    assert partial_results == [], f"第一個視窗在緩衝未滿時不應輸出，但得到 {len(partial_results)} 個"
    # 恰好到達 WINDOW_MS 時才可以輸出
    r = win.feed(config.WINDOW_MS, make_features(1.0))
    assert r is not None, "緩衝涵蓋完整 WINDOW_MS 後應輸出第一個視窗"
    print("ok  第一個視窗等到緩衝滿 WINDOW_MS 才輸出")


# ------------------------------------------------------------------ #
# 測試 2：均勻 30fps 下視窗統計值正確（mean=val, std=0, delta=0）

def test_stats_constant_signal():
    win = SlidingWindow()
    val = 42.0
    # 送 2000ms 的資料確保有視窗輸出
    results = []
    for t in range(0, 2001, 33):  # ~30fps
        r = win.feed(t, make_features(val))
        if r is not None:
            results.append(r)
    assert results, "應有視窗輸出"
    r = results[-1]
    from features import FEATURE_NAMES
    for feat in FEATURE_NAMES + ["hip_rise"]:
        assert abs(r[f"{feat}_mean"] - val) < 1e-9, f"{feat}_mean={r[f'{feat}_mean']} != {val}"
        assert abs(r[f"{feat}_std"]) < 1e-9, f"{feat}_std 應為 0"
        assert abs(r[f"{feat}_delta"]) < 1e-9, f"{feat}_delta 應為 0"
        assert abs(r[f"{feat}_max"] - val) < 1e-9
        assert abs(r[f"{feat}_min"] - val) < 1e-9
    print("ok  常數訊號：mean=val, std=0, delta=0")


# ------------------------------------------------------------------ #
# 測試 3：delta = 末幀 − 首幀（有效幀）

def test_delta_first_last_valid():
    win = SlidingWindow()
    # 送 WINDOW_MS ms 的線性遞增序列
    # 首幀 val=0, 末幀 val=WINDOW_MS（單位 ms）
    timestamps = list(range(0, config.WINDOW_MS + 1, 100))
    vals = [float(t) for t in timestamps]
    results = feed_sequence(win, timestamps, vals)
    assert results, "應有視窗輸出"
    r = results[-1]
    from features import FEATURE_NAMES
    expected_delta = vals[-1] - vals[0]
    for feat in FEATURE_NAMES + ["hip_rise"]:
        assert abs(r[f"{feat}_delta"] - expected_delta) < 1e-6, (
            f"{feat}_delta={r[f'{feat}_delta']} != {expected_delta}"
        )
    print(f"ok  delta = 末幀 − 首幀 (expected={expected_delta:.1f})")


# ------------------------------------------------------------------ #
# 測試 4：有效幀比例不足 → 回傳 None

def test_invalid_window_low_ratio():
    win = SlidingWindow()
    # 先把緩衝填滿（全部無效幀）
    # 有效幀比例 = 0 < 0.7，應回傳 None
    results = []
    for t in range(0, config.WINDOW_MS + 1, 50):
        r = win.feed(t, None)   # 全部無效
        if r is not None:
            results.append(r)
    assert results == [], f"有效幀比例不足時不應輸出，但得到 {len(results)} 個"
    print("ok  全無效幀：視窗回傳 None")


def test_invalid_window_mixed():
    """有效幀比例剛好在臨界點附近。"""
    win = SlidingWindow()
    # MIN_VALID_RATIO = 0.7
    # 送 10 幀：3 有效、7 無效 → 比例 0.3 < 0.7，應為無效視窗
    timestamps = list(range(0, config.WINDOW_MS + 1, config.WINDOW_MS // 10))
    vals = [1.0 if i < 3 else None for i in range(len(timestamps))]
    results = feed_sequence(win, timestamps, vals)
    assert results == [], f"有效幀比例 0.3 不足，不應輸出，但得到 {len(results)} 個"

    # 送 10 幀：8 有效、2 無效 → 比例 0.8 >= 0.7，應有輸出
    win2 = SlidingWindow()
    vals2 = [1.0 if i < 8 else None for i in range(len(timestamps))]
    results2 = feed_sequence(win2, timestamps, vals2)
    assert results2, "有效幀比例 0.8 足夠，應有輸出"
    print("ok  有效幀比例臨界：0.3→None，0.8→有輸出")


# ------------------------------------------------------------------ #
# 測試 5：非均勻 FPS（模擬 20fps 加抖動），視窗長度仍以 ms 計算

def test_nonuniform_fps():
    """以不均勻時間戳（模擬 20fps 有抖動）驗證視窗仍正確按 ms 切割。"""
    rng = np.random.default_rng(42)
    win = SlidingWindow()
    # 模擬 20fps：每幀約 50ms，加 ±10ms 的抖動
    total_ms = 3000
    t = 0
    timestamps = []
    while t < total_ms:
        timestamps.append(t)
        t += int(50 + rng.uniform(-10, 10))
    vals = [1.0] * len(timestamps)
    results = feed_sequence(win, timestamps, vals)
    # 預期視窗數 ≈ (total_ms - WINDOW_MS) / WINDOW_STRIDE_MS
    expected_min = (total_ms - config.WINDOW_MS) // config.WINDOW_STRIDE_MS - 2
    assert len(results) >= expected_min, (
        f"非均勻 FPS 下視窗數 {len(results)} 少於預期最低值 {expected_min}"
    )
    print(f"ok  非均勻 FPS：產生 {len(results)} 個視窗（預期 >= {expected_min}）")


# ------------------------------------------------------------------ #
# 測試 6：無效幀放入緩衝，delta 以有效幀計算

def test_delta_uses_valid_frames_only():
    """無效幀夾在中間，delta 應取最後有效幀 − 最前有效幀。

    設計：11 幀（0~1000ms, 100ms 間隔），8 幀有效（比例 8/11≈72.7% >= 70%）。
    有效幀：t=0(0.0), t=200..700(5.0), t=1000(10.0)；None：t=100, t=800, t=900。
    delta 應為 10.0 - 0.0 = 10.0（僅依有效幀計算，None 幀不影響）。
    """
    win = SlidingWindow()
    timestamps = list(range(0, 1001, 100))  # [0,100,...,1000]，共 11 幀
    # 有效 8 幀：0, 200, 300, 400, 500, 600, 700, 1000；None 3 幀：100, 800, 900
    none_set = {100, 800, 900}
    vals = []
    for t in timestamps:
        if t in none_set:
            vals.append(None)
        elif t == 0:
            vals.append(0.0)
        elif t == 1000:
            vals.append(10.0)
        else:
            vals.append(5.0)
    results = feed_sequence(win, timestamps, vals)
    assert results, "應有視窗輸出（有效幀比例 8/11≈72.7% >= 70%）"
    r = results[-1]
    from features import FEATURE_NAMES
    expected_delta = 10.0 - 0.0  # 末有效幀(10.0) − 首有效幀(0.0)
    for feat in FEATURE_NAMES + ["hip_rise"]:
        assert abs(r[f"{feat}_delta"] - expected_delta) < 1e-6, (
            f"{feat}_delta={r[f'{feat}_delta']} != {expected_delta}"
        )
    print(f"ok  delta 只用有效幀計算（expected={expected_delta:.1f}，None 幀不影響）")


# ------------------------------------------------------------------ #
# 測試 7：WINDOW_FEATURE_NAMES 長度與格式

def test_window_feature_names():
    from features import FEATURE_NAMES
    expected_count = (len(FEATURE_NAMES) + 1) * 5  # 13 特徵 × 5 統計
    assert len(WINDOW_FEATURE_NAMES) == expected_count, (
        f"WINDOW_FEATURE_NAMES 長度應為 {expected_count}，得到 {len(WINDOW_FEATURE_NAMES)}"
    )
    for name in WINDOW_FEATURE_NAMES:
        parts = name.rsplit("_", 1)
        assert len(parts) == 2 and parts[1] in ("mean", "std", "max", "min", "delta"), (
            f"欄位名格式錯誤：{name}"
        )
    print(f"ok  WINDOW_FEATURE_NAMES 共 {len(WINDOW_FEATURE_NAMES)} 個，格式正確")


# ------------------------------------------------------------------ #
# 測試 8：reset() 後視窗緩衝清空，需重新等待

def test_reset_clears_buffer():
    win = SlidingWindow()
    # 先填滿產生一個視窗
    for t in range(0, config.WINDOW_MS + 100, 50):
        win.feed(t, make_features(1.0))
    win.reset()
    # reset 後立刻送一幀，不應輸出
    r = win.feed(config.WINDOW_MS + 200, make_features(1.0))
    assert r is None, "reset 後應重新等待緩衝填滿"
    print("ok  reset() 清空緩衝，重新等待")


# ------------------------------------------------------------------ #
# 測試 9：std 使用 ddof=0（與 np.std 預設一致）

def test_std_ddof0():
    """兩個數值 [0, 2]：ddof=0 std=1.0，ddof=1 std=√2≈1.414；驗證為 ddof=0。

    設計：只送 2 幀（t=0 和 t=WINDOW_MS），全部有效（比例 2/2=100%）。
    std([0.0, 2.0], ddof=0) = 1.0；若誤用 ddof=1 則為 √2 ≈ 1.414。
    """
    win = SlidingWindow()
    timestamps = [0, config.WINDOW_MS]
    vals = [0.0, 2.0]
    results = feed_sequence(win, timestamps, vals)
    assert results, "應有視窗輸出（2 幀全有效，到達 WINDOW_MS 後輸出）"
    r = results[-1]
    from features import FEATURE_NAMES
    expected_std = 1.0  # ddof=0
    for feat in FEATURE_NAMES + ["hip_rise"]:
        assert abs(r[f"{feat}_std"] - expected_std) < 1e-9, (
            f"{feat}_std={r[f'{feat}_std']:.6f} != {expected_std} (應為 ddof=0)"
        )
    print("ok  std 使用 ddof=0（2 幀 [0,2] → std=1.0）")


# ------------------------------------------------------------------ #

if __name__ == "__main__":
    test_first_window_waits_full_duration()
    test_stats_constant_signal()
    test_delta_first_last_valid()
    test_invalid_window_low_ratio()
    test_invalid_window_mixed()
    test_nonuniform_fps()
    test_delta_uses_valid_frames_only()
    test_window_feature_names()
    test_reset_clears_buffer()
    test_std_ddof0()
    print("全部通過")
