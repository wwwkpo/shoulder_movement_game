"""分類器統一介面。

規則法與之後的 ML 分類器都繼承此 class，由 config.CLASSIFIER 切換。
"""


class Classifier:
    def predict(self, window_features: dict) -> tuple[str, float]:
        """
        輸入：window_features dict（WINDOW_FEATURE_NAMES 的 65 維視窗統計特徵）
        回傳：(label, prob)
            label：11 個類別之一（left_45/90/135/180、right_45/90/135/180、squat、jump、none）
            prob：信心值（規則法固定回傳 1.0；ML 分類器回傳真實機率）
        """
        raise NotImplementedError
