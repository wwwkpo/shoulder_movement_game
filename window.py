"""滑動視窗：把逐幀特徵彙整成統計特徵供分類器使用。

視窗定義
--------
- 長度：WINDOW_MS 毫秒（預設 1000ms）
- 步距：WINDOW_STRIDE_MS 毫秒（預設 200ms），每隔此時間輸出一個視窗
- 第一個視窗必須等到緩衝涵蓋完整 WINDOW_MS 才輸出

有效幀判斷
----------
- 無效幀（features=None）也要放入緩衝，佔一個位置
- 有效幀比例 = 有效幀數 / 視窗內總幀數
- 比例 < MIN_VALID_RATIO → 該視窗無效，回傳 None，不送分類器

統計量（每個特徵各算五個）
--------------------------
- mean：有效幀的算術平均
- std：有效幀的標準差（ddof=0，即 np.std 預設值）
- max：有效幀最大值
- min：有效幀最小值
- delta：視窗內「最後一個有效幀」- 「第一個有效幀」

特徵欄位共 13 × 5 = 65 維，欄位名格式 {feature}_{stat}，
例如 L_shoulder_angle_mean。欄位順序固定於 WINDOW_FEATURE_NAMES。

ML 訓練資料與規則法使用同一份 window_features，不得改動此格式。
"""
from collections import deque

import numpy as np

import config
from features import FEATURE_NAMES

# 13 個逐幀特徵（12 個基本 + hip_rise，由 CalibrationResult.add_hip_rise 加入）
_FRAME_FEATURES = FEATURE_NAMES + ["hip_rise"]
_STATS = ["mean", "std", "max", "min", "delta"]

# 視窗特徵欄位順序（固定，ML 訓練資料與規則法共用）
WINDOW_FEATURE_NAMES = [
    f"{feat}_{stat}"
    for feat in _FRAME_FEATURES
    for stat in _STATS
]


def _compute_stats(values: list[float]) -> dict[str, float]:
    """對一組有效幀數值算五個統計量。values 長度保證 >= 1。"""
    arr = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),       # ddof=0
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "delta": float(arr[-1] - arr[0]),  # 最後有效幀 − 最先有效幀
    }


class SlidingWindow:
    """
    逐幀 feed 資料，定期（每 WINDOW_STRIDE_MS）輸出一個視窗特徵 dict。

    用法：
        win = SlidingWindow()
        for frame, ts in ...:
            result = win.feed(ts, features_with_hip_rise)
            # result 為 dict（有效視窗）或 None（無效視窗或尚未到輸出時間）
    """

    def __init__(self):
        # 緩衝：每項為 (timestamp_ms, features_dict_or_None)
        self._buffer: deque[tuple[int, dict | None]] = deque()
        self._next_output_ts: int | None = None   # 下一次該輸出視窗的時間戳
        self._start_ts: int | None = None          # 第一幀的時間戳（_trim 後 earliest_ts 會變動，需另行記錄）

    # ------------------------------------------------------------------ #
    def feed(self, timestamp_ms: int, features: dict | None) -> dict | None:
        """
        餵入一幀，回傳視窗特徵 dict 或 None。
        features 為 None 表示該幀無效（遮擋 / 關鍵點不足），仍需放入緩衝。
        """
        if self._start_ts is None:
            self._start_ts = timestamp_ms

        self._buffer.append((timestamp_ms, features))
        self._trim(timestamp_ms)

        # 決定是否該輸出
        if self._next_output_ts is None:
            # 第一次：等到距第一幀已累積完整 WINDOW_MS 才開始輸出
            # 注意：不能用 buffer[0][0] 判斷，因為 _trim 後 earliest_ts 已前移
            if timestamp_ms - self._start_ts < config.WINDOW_MS:
                return None
            self._next_output_ts = timestamp_ms

        if timestamp_ms < self._next_output_ts:
            return None

        self._next_output_ts = timestamp_ms + config.WINDOW_STRIDE_MS
        return self._compute_window()

    # ------------------------------------------------------------------ #
    def _trim(self, current_ts: int):
        """移除超出視窗長度的舊幀。"""
        cutoff = current_ts - config.WINDOW_MS
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    # ------------------------------------------------------------------ #
    def _compute_window(self) -> dict | None:
        """
        計算目前緩衝的視窗特徵。
        有效幀比例 < MIN_VALID_RATIO → 回傳 None。
        """
        total = len(self._buffer)
        valid_frames = [(ts, f) for ts, f in self._buffer if f is not None]
        n_valid = len(valid_frames)

        if total == 0 or n_valid / total < config.MIN_VALID_RATIO:
            return None

        # 對每個特徵收集有效幀的值（按時間順序，delta 才能正確算首-末）
        feature_values: dict[str, list[float]] = {feat: [] for feat in _FRAME_FEATURES}
        for _, f in valid_frames:
            for feat in _FRAME_FEATURES:
                val = f.get(feat)
                if val is not None:
                    feature_values[feat].append(float(val))

        result = {}
        for feat in _FRAME_FEATURES:
            vals = feature_values[feat]
            if not vals:
                # 該特徵所有有效幀都缺失（罕見，但要能處理）
                for stat in _STATS:
                    result[f"{feat}_{stat}"] = 0.0
            else:
                stats = _compute_stats(vals)
                for stat in _STATS:
                    result[f"{feat}_{stat}"] = stats[stat]

        return result

    # ------------------------------------------------------------------ #
    def reset(self):
        """清空緩衝（換題時呼叫）。"""
        self._buffer.clear()
        self._next_output_ts = None
        self._start_ts = None
