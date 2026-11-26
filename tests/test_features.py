"""用人工合成的骨架驗證特徵計算，不需要鏡頭也不需要模型檔。

執行：python tests/test_features.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from features import compute_frame_features  # noqa: E402
from pose import (  # noqa: E402
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)

ARM = 80.0  # 上臂、前臂長度（像素）


def rotate(v, deg):
    r = math.radians(deg)
    return np.array([v[0] * math.cos(r) - v[1] * math.sin(r),
                     v[0] * math.sin(r) + v[1] * math.cos(r)])


def make_pose(l_arm=10.0, r_arm=10.0, hip_drop=0.0, mirror=False, width=1280):
    """
    合成一個面向鏡頭站立的人（未鏡像：本人左邊在影像右邊）。
    l_arm / r_arm：手臂與軀幹的夾角（度），手臂打直、往身體外側舉。
    hip_drop：髖部往下移的像素（模擬蹲下）。
    """
    lm = np.zeros((33, 4))
    lm[:, 3] = 1.0
    pts = {
        R_SHOULDER: (600, 300), L_SHOULDER: (700, 300),
        R_HIP: (615, 500 + hip_drop), L_HIP: (685, 500 + hip_drop),
        # 蹲下時膝蓋往外張（正面看，膝蓋往鏡頭彎的部分在 2D 看不到）
        R_KNEE: (612 - hip_drop * 0.5, 600 + hip_drop / 2),
        L_KNEE: (688 + hip_drop * 0.5, 600 + hip_drop / 2),
        R_ANKLE: (610, 700), L_ANKLE: (690, 700),
    }
    for idx, (x, y) in pts.items():
        lm[idx, :2] = (x, y)

    for sh, hip, el, wr, ang, outward in [
        (L_SHOULDER, L_HIP, L_ELBOW, L_WRIST, l_arm, +1),
        (R_SHOULDER, R_HIP, R_ELBOW, R_WRIST, r_arm, -1),
    ]:
        down = lm[hip, :2] - lm[sh, :2]
        down = down / np.linalg.norm(down)
        # 兩個旋轉方向中，選 x 分量比較往身體外側的那個
        d1, d2 = rotate(down, ang), rotate(down, -ang)
        direction = d1 if d1[0] * outward >= d2[0] * outward else d2
        lm[el, :2] = lm[sh, :2] + direction * ARM
        lm[wr, :2] = lm[el, :2] + direction * ARM

    if mirror:
        lm[:, 0] = width - lm[:, 0]
    return lm


def approx(a, b, tol=1.0):
    return abs(a - b) <= tol


def test_shoulder_angles():
    for ang in (45, 90, 135, 180):
        f, _, missing = compute_frame_features(make_pose(l_arm=ang, r_arm=20))
        assert not missing, missing
        assert approx(f["L_shoulder_angle"], ang), (ang, f["L_shoulder_angle"])
        assert approx(f["R_shoulder_angle"], 20), f["R_shoulder_angle"]
        assert approx(f["L_elbow_angle"], 180), f["L_elbow_angle"]
        f, _, _ = compute_frame_features(make_pose(l_arm=20, r_arm=ang))
        assert approx(f["R_shoulder_angle"], ang), (ang, f["R_shoulder_angle"])
    print("ok  肩角 45/90/135/180 左右手都正確")


def test_wrist_outward_positive():
    f, _, _ = compute_frame_features(make_pose(l_arm=90, r_arm=90))
    assert f["L_wrist_dx"] > 1.0 and f["R_wrist_dx"] > 1.0, (f["L_wrist_dx"], f["R_wrist_dx"])
    assert abs(f["L_wrist_dy"]) < 0.3 and abs(f["R_wrist_dy"]) < 0.3
    f, _, _ = compute_frame_features(make_pose(l_arm=180, r_arm=10))
    assert f["L_wrist_dy"] > 1.5, f["L_wrist_dy"]
    print("ok  手腕相對位置：往外為正、舉高 dy 為正")


def test_mirror_invariant():
    a, _, _ = compute_frame_features(make_pose(l_arm=135, r_arm=45))
    b, _, _ = compute_frame_features(make_pose(l_arm=135, r_arm=45, mirror=True))
    for k in a:
        assert approx(a[k], b[k], 1e-6), (k, a[k], b[k])
    print("ok  影像左右翻轉後特徵完全相同（左右手不會對調）")


def test_squat_lowers_hip():
    stand, raw_s, _ = compute_frame_features(make_pose())
    squat, raw_q, _ = compute_frame_features(make_pose(hip_drop=100))
    assert squat["hip_ankle_height"] < stand["hip_ankle_height"] - 0.5
    assert squat["L_knee_angle"] < stand["L_knee_angle"] - 20
    assert raw_q["hip_y_px"] > raw_s["hip_y_px"]
    print(f"ok  蹲下：hip_ankle_height {stand['hip_ankle_height']:.2f} -> "
          f"{squat['hip_ankle_height']:.2f}，膝角 {stand['L_knee_angle']:.0f} -> "
          f"{squat['L_knee_angle']:.0f}")


def test_low_visibility_rejected():
    lm = make_pose()
    lm[L_ANKLE, 3] = 0.1
    f, raw, missing = compute_frame_features(lm)
    assert f is None and raw is None and missing == ["L_ankle"], missing
    print("ok  關鍵點看不到時不計算特徵，並回報缺哪個點")


if __name__ == "__main__":
    test_shoulder_angles()
    test_wrist_outward_positive()
    test_mirror_invariant()
    test_squat_lowers_hip()
    test_low_visibility_rejected()
    print("全部通過")
