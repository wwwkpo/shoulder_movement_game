"""站立校正：開局收集 CALIBRATION_SEC 秒的站立幀，取中位數作為基準。

校正失敗條件：
1. 有效幀比例 < CALIBRATION_MIN_VALID_RATIO（遮擋太多）
2. hip_y_px 標準差 / 肩寬中位數 > CALIBRATION_MAX_HIP_STD_RATIO（沒站穩）

校正結果會新增第 13 個逐幀特徵：
    hip_rise = (base_hip_y_px - hip_y_px) / base_shoulder_width_px
    （往上為正，跳躍時為正值，蹲下時為負值）
"""
import numpy as np

import config


class CalibrationResult:
    """校正基準值容器，同時作為特徵補充器。"""

    def __init__(self, base_hip_y_px, base_shoulder_width_px, base_hip_ankle_height):
        self.base_hip_y_px = float(base_hip_y_px)
        self.base_shoulder_width_px = float(base_shoulder_width_px)
        self.base_hip_ankle_height = float(base_hip_ankle_height)

    def add_hip_rise(self, features: dict, raw: dict) -> dict:
        """回傳加入 hip_rise 後的新 features dict（不修改原始 dict）。"""
        hip_rise = (self.base_hip_y_px - raw["hip_y_px"]) / self.base_shoulder_width_px
        return {**features, "hip_rise": hip_rise}

    def shoulder_width_ok(self, raw: dict) -> bool:
        """肩寬與基準相差不超過 SHOULDER_WIDTH_DRIFT_RATIO 才算站在原位。"""
        ratio = (abs(raw["shoulder_width_px"] - self.base_shoulder_width_px)
                 / self.base_shoulder_width_px)
        return ratio <= config.SHOULDER_WIDTH_DRIFT_RATIO


class Calibrator:
    """校正流程管理器。

    用法（在主迴圈裡）：
        cal = Calibrator()
        while cal.collecting:
            frame, ts = source.read()
            landmarks = estimator.detect(frame, ts)
            if landmarks is not None:
                features, raw, missing = compute_frame_features(landmarks)
            else:
                features, raw = None, None
            msg, done = cal.feed(ts, features, raw)
            # msg 是給使用者看的提示字串
            # done=True 時若 cal.success → 用 cal.result；若 cal.failed → 重新 Calibrator()
    """

    def __init__(self):
        self.collecting = True   # False 表示校正結束
        self.success = False
        self.failed = False
        self._result: CalibrationResult | None = None

        self._start_ts: int | None = None
        # 有效幀分別存 raw（hip_y_px、shoulder_width_px）與 features（需要 hip_ankle_height）
        self._valid_raw: list[dict] = []
        self._valid_features: list[dict] = []
        self._total_frames: int = 0

    # ------------------------------------------------------------------ #
    @property
    def result(self) -> CalibrationResult | None:
        return self._result

    # ------------------------------------------------------------------ #
    def feed(self, timestamp_ms: int, features, raw) -> tuple[str, bool]:
        """
        餵入一幀的特徵資料。
        回傳 (提示文字, 是否結束)。
        結束後 self.success=True → 呼叫 self.result；self.failed=True → 重新校正。
        """
        if not self.collecting:
            return "", True

        if self._start_ts is None:
            self._start_ts = timestamp_ms

        elapsed_ms = timestamp_ms - self._start_ts
        remaining_s = max(0.0, config.CALIBRATION_SEC - elapsed_ms / 1000.0)

        self._total_frames += 1
        if features is not None and raw is not None:
            self._valid_raw.append(raw.copy())
            self._valid_features.append(features.copy())

        # 還在收集中
        if elapsed_ms < config.CALIBRATION_SEC * 1000:
            return f"請自然站立，校正中… {remaining_s:.1f}s", False

        # 收集結束
        return self._finalize()

    # ------------------------------------------------------------------ #
    def _finalize(self) -> tuple[str, bool]:
        """收集完畢後計算基準值，回傳 (訊息, True)。"""
        self.collecting = False
        n_valid = len(self._valid_raw)
        n_total = self._total_frames

        if n_total == 0 or n_valid / n_total < config.CALIBRATION_MIN_VALID_RATIO:
            self.failed = True
            return (
                f"校正失敗：有效幀不足（{n_valid}/{n_total}），請確認全身入鏡後重試。",
                True,
            )

        hip_ys = np.array([f["hip_y_px"] for f in self._valid_raw])
        shoulder_ws = np.array([f["shoulder_width_px"] for f in self._valid_raw])
        median_sw = float(np.median(shoulder_ws))

        # 沒站穩判斷
        hip_std_ratio = float(np.std(hip_ys)) / median_sw
        if hip_std_ratio > config.CALIBRATION_MAX_HIP_STD_RATIO:
            self.failed = True
            return (
                f"校正失敗：身體晃動太大（hip_std/肩寬={hip_std_ratio:.3f}>{config.CALIBRATION_MAX_HIP_STD_RATIO}），請站穩後重試。",
                True,
            )

        base_hip_ankle = float(np.median(
            [f["hip_ankle_height"] for f in self._valid_features]
        ))

        self.success = True
        self._result = CalibrationResult(
            base_hip_y_px=float(np.median(hip_ys)),
            base_shoulder_width_px=median_sw,
            base_hip_ankle_height=base_hip_ankle,
        )
        return "校正成功！", True
