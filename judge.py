"""判定機制：連續視窗確認。

連續 CONFIRM_WINDOWS 個有效視窗預測同一類、且 prob > PROB_THRESHOLD
→ 判定成立，feed() 回傳 (label, confirm_ts)（同一 streak 僅回傳一次）。

無效視窗（label=None）會打斷連續計數。
換題時呼叫 reset()。
"""
import config


class Judge:
    def __init__(self):
        self._streak_label: str | None = None
        self._streak_count: int = 0
        self._confirmed: bool = False   # 當前 streak 是否已觸發過確認

    # ------------------------------------------------------------------ #
    def feed(self, label: str | None, prob: float, ts: int) -> tuple[str, int] | None:
        """
        餵入一個視窗的預測結果。
        label=None 表示該視窗無效（win.feed() 回傳 None，不送分類器）。

        回傳值：
          (label, ts)  判定剛成立（同一 streak 僅火一次）
          None         尚未達到判定條件
        """
        # 無效視窗或機率不超過門檻 → 打斷連續計數
        if label is None or prob <= config.PROB_THRESHOLD:
            self._reset_streak()
            return None

        # 與當前 streak label 不同 → 重設並從 1 重計
        if label != self._streak_label:
            self._reset_streak()
            self._streak_label = label
            self._streak_count = 1
        else:
            # 同 label，繼續累積
            self._streak_count += 1

        if self._streak_count >= config.CONFIRM_WINDOWS and not self._confirmed:
            self._confirmed = True
            return (label, ts)

        return None

    # ------------------------------------------------------------------ #
    def reset(self):
        """清空連續計數（換題或重新校正時呼叫）。"""
        self._reset_streak()

    # ------------------------------------------------------------------ #
    def _reset_streak(self):
        self._streak_label = None
        self._streak_count = 0
        self._confirmed = False
