"""test_judge.py：Judge 判定機制的單元測試。

執行：python tests/test_judge.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from judge import Judge


class TestJudge(unittest.TestCase):

    def setUp(self):
        """每個測試前備份可調參數，確保測試間互不影響。"""
        self._orig_confirm = config.CONFIRM_WINDOWS
        self._orig_prob    = config.PROB_THRESHOLD
        config.CONFIRM_WINDOWS = 3
        config.PROB_THRESHOLD  = 0.7

    def tearDown(self):
        config.CONFIRM_WINDOWS = self._orig_confirm
        config.PROB_THRESHOLD  = self._orig_prob

    # ------------------------------------------------------------------ #
    # 正常確認
    # ------------------------------------------------------------------ #

    def test_confirm_after_3_windows(self):
        """連續 3 個有效視窗相同 label → 判定成立，回傳 (label, ts)。"""
        j = Judge()
        self.assertIsNone(j.feed("squat", 1.0, 100))
        self.assertIsNone(j.feed("squat", 1.0, 200))
        result = j.feed("squat", 1.0, 300)
        self.assertEqual(result, ("squat", 300))

    def test_confirm_returns_correct_ts(self):
        """確認成立時的 ts 應等於第 CONFIRM_WINDOWS 次視窗的 ts。"""
        j = Judge()
        j.feed("jump", 1.0, 500)
        j.feed("jump", 1.0, 700)
        result = j.feed("jump", 1.0, 900)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], 900)

    # ------------------------------------------------------------------ #
    # 無效視窗打斷
    # ------------------------------------------------------------------ #

    def test_none_label_breaks_streak(self):
        """無效視窗（label=None）打斷連續計數，之後需重新累積。"""
        j = Judge()
        j.feed("left_90", 1.0, 100)
        j.feed("left_90", 1.0, 200)
        j.feed(None, 0.0, 300)          # 無效視窗
        self.assertIsNone(j.feed("left_90", 1.0, 400))  # streak=1，不夠

    def test_none_breaks_then_recount(self):
        """無效後需再滿 CONFIRM_WINDOWS 次才能確認。"""
        j = Judge()
        j.feed("squat", 1.0, 100)
        j.feed("squat", 1.0, 200)
        j.feed(None, 0.0, 300)
        j.feed("squat", 1.0, 400)
        j.feed("squat", 1.0, 500)
        result = j.feed("squat", 1.0, 600)
        self.assertEqual(result, ("squat", 600))

    # ------------------------------------------------------------------ #
    # 不同 label 打斷
    # ------------------------------------------------------------------ #

    def test_different_label_breaks_streak(self):
        """不同 label 重設 streak，新 label 從第 1 次重計。"""
        j = Judge()
        j.feed("left_90", 1.0, 100)
        j.feed("left_90", 1.0, 200)
        j.feed("squat", 1.0, 300)       # 不同 label，重設
        self.assertIsNone(j.feed("squat", 1.0, 400))   # streak=2
        result = j.feed("squat", 1.0, 500)              # streak=3
        self.assertEqual(result, ("squat", 500))

    # ------------------------------------------------------------------ #
    # 機率門檻
    # ------------------------------------------------------------------ #

    def test_prob_at_threshold_not_counted(self):
        """prob == PROB_THRESHOLD（不超過）不計入 streak。"""
        j = Judge()
        j.feed("none", 0.7, 100)   # 恰好等於，不超過
        j.feed("none", 0.7, 200)
        result = j.feed("none", 0.7, 300)
        self.assertIsNone(result)

    def test_prob_below_threshold_breaks_streak(self):
        """機率不足的視窗打斷 streak。"""
        j = Judge()
        j.feed("jump", 1.0, 100)
        j.feed("jump", 1.0, 200)
        j.feed("jump", 0.5, 300)   # 機率不足
        result = j.feed("jump", 1.0, 400)
        self.assertIsNone(result)  # streak 被打斷，只有 1

    # ------------------------------------------------------------------ #
    # 確認只觸發一次
    # ------------------------------------------------------------------ #

    def test_confirm_fires_only_once(self):
        """同一 streak 確認成立後，繼續餵相同 label 不再回傳 (label, ts)。"""
        j = Judge()
        j.feed("right_45", 1.0, 100)
        j.feed("right_45", 1.0, 200)
        first  = j.feed("right_45", 1.0, 300)   # 第一次觸發
        second = j.feed("right_45", 1.0, 400)   # 不再觸發
        third  = j.feed("right_45", 1.0, 500)
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIsNone(third)

    # ------------------------------------------------------------------ #
    # reset()
    # ------------------------------------------------------------------ #

    def test_reset_clears_streak(self):
        """reset() 清空連續計數，reset 後需從頭累積。"""
        j = Judge()
        j.feed("squat", 1.0, 100)
        j.feed("squat", 1.0, 200)
        j.reset()
        result = j.feed("squat", 1.0, 300)   # reset 後 streak=1
        self.assertIsNone(result)

    def test_confirm_possible_after_reset(self):
        """reset() 後可重新累積並確認另一個 label。"""
        j = Judge()
        j.feed("squat", 1.0, 100)
        j.feed("squat", 1.0, 200)
        j.feed("squat", 1.0, 300)   # 第一次確認
        j.reset()
        j.feed("jump", 1.0, 400)
        j.feed("jump", 1.0, 500)
        result = j.feed("jump", 1.0, 600)
        self.assertEqual(result, ("jump", 600))

    # ------------------------------------------------------------------ #
    # CONFIRM_WINDOWS=1 邊界
    # ------------------------------------------------------------------ #

    def test_confirm_windows_1(self):
        """CONFIRM_WINDOWS=1 時，第一個有效視窗就觸發確認。"""
        config.CONFIRM_WINDOWS = 1
        j = Judge()
        result = j.feed("left_45", 1.0, 100)
        self.assertEqual(result, ("left_45", 100))


if __name__ == "__main__":
    unittest.main(verbosity=2)
