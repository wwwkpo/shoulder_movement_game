"""規則法分類器。

判斷優先順序：jump → squat → 手臂 → none
所有門檻都在 config.py，程式內不寫魔術數字。

predict() 介面符合 classifiers/base.py，機率固定回傳 1.0。
explain() 回傳各條件的實際值與是否通過，方便實測時顯示在畫面上調門檻。
"""
import config
from classifiers.base import Classifier


def _angle_grade(angle: float) -> int:
    """把肩角映射到 45/90/135/180 四級（取最接近的分界）。"""
    boundaries = config.ARM_ANGLE_BOUNDARIES  # (67.5, 112.5, 157.5)
    if angle < boundaries[0]:
        return 45
    elif angle < boundaries[1]:
        return 90
    elif angle < boundaries[2]:
        return 135
    else:
        return 180


class RuleClassifier(Classifier):

    def predict(self, window_features: dict) -> tuple[str, float]:
        """
        回傳 (label, 1.0)。
        label 為 11 個類別之一：
            left_45/90/135/180、right_45/90/135/180、squat、jump、none
        """
        label = self._classify(window_features)
        return label, 1.0

    # ------------------------------------------------------------------ #
    def explain(self, window_features: dict) -> list[str]:
        """
        回傳各條件的實際值與是否通過的說明字串清單，格式範例：
            ["hip_rise_max=0.12 < 0.25 ✗  (jump 未觸發)",
             "hip_rise_mean=-0.03 > -0.5 ✗  (squat 未觸發)",
             "L_shoulder_angle_mean=87.3 >= 25 ✓",
             "L_elbow_angle_mean=172.1 >= 140 ✓",
             "L_shoulder_angle_std=4.2 <= 12 ✓",
             "=> left_90"]
        """
        wf = window_features
        lines = []

        # --- jump ---
        rise_max = wf.get("hip_rise_max", 0.0)
        rise_min = wf.get("hip_rise_min", 0.0)
        rise_range = rise_max - rise_min
        j1 = rise_max > config.JUMP_RISE
        j2 = rise_range > config.JUMP_RANGE
        lines.append(
            f"hip_rise_max={rise_max:.3f} > {config.JUMP_RISE} {'✓' if j1 else '✗'}"
        )
        lines.append(
            f"hip_rise_range={rise_range:.3f} > {config.JUMP_RANGE} {'✓' if j2 else '✗'}"
        )

        # --- squat ---
        rise_mean = wf.get("hip_rise_mean", 0.0)
        sq = rise_mean < config.SQUAT_DROP
        lines.append(
            f"hip_rise_mean={rise_mean:.3f} < {config.SQUAT_DROP} {'✓' if sq else '✗'}"
        )

        # --- arms ---
        for side, prefix in [("L", "L"), ("R", "R")]:
            ang_mean = wf.get(f"{prefix}_shoulder_angle_mean", 0.0)
            elb_mean = wf.get(f"{prefix}_elbow_angle_mean", 0.0)
            ang_std = wf.get(f"{prefix}_shoulder_angle_std", 999.0)
            c1 = ang_mean >= config.ARM_RAISED_MIN
            c2 = elb_mean >= config.ELBOW_STRAIGHT_MIN
            c3 = ang_std <= config.ARM_STABLE_STD
            lines.append(
                f"{prefix}_sh_ang_mean={ang_mean:.1f}>={config.ARM_RAISED_MIN} {'✓' if c1 else '✗'} "
                f"elb={elb_mean:.1f}>={config.ELBOW_STRAIGHT_MIN} {'✓' if c2 else '✗'} "
                f"std={ang_std:.1f}<={config.ARM_STABLE_STD} {'✓' if c3 else '✗'}"
            )

        label = self._classify(window_features)
        lines.append(f"=> {label}")
        return lines

    # ------------------------------------------------------------------ #
    def _classify(self, wf: dict) -> str:
        """內部分類邏輯，回傳 label 字串。"""

        # 1. 跳躍
        rise_max = wf.get("hip_rise_max", 0.0)
        rise_min = wf.get("hip_rise_min", 0.0)
        if rise_max > config.JUMP_RISE and (rise_max - rise_min) > config.JUMP_RANGE:
            return "jump"

        # 2. 蹲下
        rise_mean = wf.get("hip_rise_mean", 0.0)
        if rise_mean < config.SQUAT_DROP:
            return "squat"

        # 3. 手臂
        def arm_raised(prefix: str) -> bool:
            ang_mean = wf.get(f"{prefix}_shoulder_angle_mean", 0.0)
            elb_mean = wf.get(f"{prefix}_elbow_angle_mean", 0.0)
            ang_std = wf.get(f"{prefix}_shoulder_angle_std", 999.0)
            return (ang_mean >= config.ARM_RAISED_MIN
                    and elb_mean >= config.ELBOW_STRAIGHT_MIN
                    and ang_std <= config.ARM_STABLE_STD)

        l_up = arm_raised("L")
        r_up = arm_raised("R")

        # 兩手同時舉起 → none（遊戲規定一次一隻手）
        if l_up and r_up:
            return "none"

        if l_up:
            grade = _angle_grade(wf.get("L_shoulder_angle_mean", 0.0))
            return f"left_{grade}"

        if r_up:
            grade = _angle_grade(wf.get("R_shoulder_angle_mean", 0.0))
            return f"right_{grade}"

        # 4. 其他
        return "none"
