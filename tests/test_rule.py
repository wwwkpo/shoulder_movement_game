"""驗證規則分類器能正確分類 11 個類別，以及邊界情況。

執行：python tests/test_rule.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from classifiers.rule import RuleClassifier
from window import SlidingWindow, WINDOW_FEATURE_NAMES
from features import FEATURE_NAMES, compute_frame_features
from calibration import CalibrationResult
from tests.test_features import make_pose


# ------------------------------------------------------------------ #
# 輔助：用合成骨架序列建立視窗特徵

def build_window_features(
    l_arm=10.0, r_arm=10.0, hip_drop=0.0,
    n_frames=30, hip_rise_override=None
) -> dict | None:
    """
    建立 n_frames 幀的合成序列，回傳最後一個有效視窗特徵。
    hip_rise_override：直接指定每幀的 hip_rise 值（模擬跳躍/蹲下）。
    """
    # 建立校正基準
    lm_stand = make_pose()
    _, raw_stand, _ = compute_frame_features(lm_stand)
    calib = CalibrationResult(
        base_hip_y_px=raw_stand["hip_y_px"],
        base_shoulder_width_px=raw_stand["shoulder_width_px"],
        base_hip_ankle_height=0.0,
    )

    win = SlidingWindow()
    # 送足夠幀（WINDOW_MS + 額外 500ms）
    total_ms = config.WINDOW_MS + 500
    dt = total_ms // n_frames
    last_result = None
    for i in range(n_frames):
        ts = i * dt
        lm = make_pose(l_arm=l_arm, r_arm=r_arm, hip_drop=hip_drop)
        feat, raw, missing = compute_frame_features(lm)
        if feat is None:
            r = win.feed(ts, None)
        else:
            feat_with_rise = calib.add_hip_rise(feat, raw)
            # 允許外部覆蓋 hip_rise（模擬跳躍/蹲下）
            if hip_rise_override is not None:
                feat_with_rise = {**feat_with_rise, "hip_rise": hip_rise_override(i, n_frames)}
            r = win.feed(ts, feat_with_rise)
        if r is not None:
            last_result = r
    return last_result


def classify(wf: dict) -> str:
    clf = RuleClassifier()
    label, prob = clf.predict(wf)
    assert prob == 1.0, f"規則法機率應固定 1.0，得到 {prob}"
    return label


# ------------------------------------------------------------------ #
# 測試 1：無動作 → none

def test_none():
    wf = build_window_features(l_arm=10, r_arm=10)
    assert wf is not None
    assert classify(wf) == "none", f"無動作應為 none，得 {classify(wf)}"
    print("ok  無動作 → none")


# ------------------------------------------------------------------ #
# 測試 2：左手四個角度

def test_left_arm_angles():
    for target_deg in (45, 90, 135, 180):
        wf = build_window_features(l_arm=target_deg, r_arm=10)
        assert wf is not None
        label = classify(wf)
        expected = f"left_{target_deg}"
        assert label == expected, f"左手 {target_deg}° 應為 {expected}，得 {label}"
    print("ok  左手 45/90/135/180 分類正確")


# ------------------------------------------------------------------ #
# 測試 3：右手四個角度

def test_right_arm_angles():
    for target_deg in (45, 90, 135, 180):
        wf = build_window_features(l_arm=10, r_arm=target_deg)
        assert wf is not None
        label = classify(wf)
        expected = f"right_{target_deg}"
        assert label == expected, f"右手 {target_deg}° 應為 {expected}，得 {label}"
    print("ok  右手 45/90/135/180 分類正確")


# ------------------------------------------------------------------ #
# 測試 4：兩手同時舉起 → none

def test_both_arms_raised():
    wf = build_window_features(l_arm=90, r_arm=90)
    assert wf is not None
    assert classify(wf) == "none", f"兩手同舉應為 none，得 {classify(wf)}"
    print("ok  兩手同舉 → none")


# ------------------------------------------------------------------ #
# 測試 5：手臂移動中（std 大）→ none

def test_arm_moving_std_too_high():
    """模擬手臂在 30°~150° 之間快速擺動，std 應超過 ARM_STABLE_STD。"""
    calib = CalibrationResult(
        base_hip_y_px=500.0,
        base_shoulder_width_px=100.0,
        base_hip_ankle_height=0.0,
    )
    win = SlidingWindow()
    total_ms = config.WINDOW_MS + 500
    n = 40
    dt = total_ms // n
    last_result = None
    for i in range(n):
        ts = i * dt
        # 肩角在 30~150 之間振盪
        ang = 30.0 + 120.0 * (i % 2)
        lm = make_pose(l_arm=ang, r_arm=10)
        feat, raw, _ = compute_frame_features(lm)
        if feat is not None:
            feat_with_rise = calib.add_hip_rise(feat, raw)
            r = win.feed(ts, feat_with_rise)
        else:
            r = win.feed(ts, None)
        if r is not None:
            last_result = r
    assert last_result is not None
    # std 超過閾值時不應判定為手臂
    assert classify(last_result) == "none", (
        f"手臂擺動中（std 大）應為 none，得 {classify(last_result)}, "
        f"L_shoulder_angle_std={last_result.get('L_shoulder_angle_std', '?'):.2f}"
    )
    print(f"ok  手臂擺動中 → none（L_shoulder_angle_std="
          f"{last_result.get('L_shoulder_angle_std', 0):.2f} > {config.ARM_STABLE_STD}）")


# ------------------------------------------------------------------ #
# 測試 6：跳躍

def test_jump():
    """覆蓋 hip_rise 模擬跳躍上下起伏。"""
    def hip_rise_jump(i, n):
        # 前半段上升（hip_rise > JUMP_RISE），後半段回到 0
        # max = 0.4, min = 0
        half = n // 2
        return 0.4 if i < half else 0.0

    wf = build_window_features(hip_rise_override=hip_rise_jump)
    assert wf is not None
    # hip_rise_max=0.4 > 0.25, range=0.4 > 0.35 → jump
    assert classify(wf) == "jump", (
        f"跳躍應為 jump，得 {classify(wf)}，"
        f"hip_rise_max={wf.get('hip_rise_max', '?')}, "
        f"hip_rise_min={wf.get('hip_rise_min', '?')}"
    )
    print("ok  跳躍 → jump")


# ------------------------------------------------------------------ #
# 測試 7：蹲下

def test_squat():
    """覆蓋 hip_rise 模擬蹲下（hip_rise_mean < SQUAT_DROP=-0.5）。"""
    def hip_rise_squat(i, n):
        return -0.7  # 持續低於 SQUAT_DROP

    wf = build_window_features(hip_rise_override=hip_rise_squat)
    assert wf is not None
    assert classify(wf) == "squat", (
        f"蹲下應為 squat，得 {classify(wf)}，"
        f"hip_rise_mean={wf.get('hip_rise_mean', '?')}"
    )
    print("ok  蹲下 → squat")


# ------------------------------------------------------------------ #
# 測試 8：跳躍優先於手臂

def test_jump_priority_over_arm():
    """同時有跳躍信號和手臂舉起，jump 應優先。"""
    def hip_rise_jump(i, n):
        half = n // 2
        return 0.4 if i < half else 0.0

    # 左手 90°，同時 hip_rise 有跳躍模式
    wf = build_window_features(l_arm=90, hip_rise_override=hip_rise_jump)
    assert wf is not None
    label = classify(wf)
    assert label == "jump", f"跳躍優先於手臂，應為 jump，得 {label}"
    print("ok  跳躍優先於手臂舉起")


# ------------------------------------------------------------------ #
# 測試 9：explain() 回傳格式與最後一行為 => label

def test_explain_format():
    clf = RuleClassifier()
    wf = build_window_features(l_arm=90, r_arm=10)
    assert wf is not None
    lines = clf.explain(wf)
    assert isinstance(lines, list) and len(lines) > 0, "explain() 應回傳非空清單"
    assert lines[-1].startswith("=> "), f"最後一行應以 '=> ' 開頭，得 {lines[-1]!r}"
    # 確認有 ✓ 或 ✗
    all_text = " ".join(lines)
    assert "✓" in all_text or "✗" in all_text, "explain() 應包含 ✓ 或 ✗"
    print(f"ok  explain() 格式正確，共 {len(lines)} 行，結果：{lines[-1]}")


# ------------------------------------------------------------------ #

if __name__ == "__main__":
    test_none()
    test_left_arm_angles()
    test_right_arm_angles()
    test_both_arms_raised()
    test_arm_moving_std_too_high()
    test_jump()
    test_squat()
    test_jump_priority_over_arm()
    test_explain_format()
    print("全部通過")
