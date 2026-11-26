# 課間 3 分鐘肩部伸展體感遊戲 — 雛型開發計畫（規則法版）

> 給 Claude Code 的執行計畫。依里程碑順序實作，每個里程碑完成後跑測試並停下來，
> 列出「需要使用者用 Webcam 實測確認的項目」，等使用者回報後再進行下一個。

## 0. 背景與範圍

課程「影像辨識應用及動作辨識導論」期中專題。用筆電 Webcam + MediaPipe Pose 辨識手臂側舉角度與全身動作，做成「依提示做出指定伸展姿勢」的體感遊戲。

**本計畫範圍：只做規則法雛型，確認整個遊戲能正常運作。** 機器學習之後再加，範圍只有 Logistic Regression 與 Random Forest（不做 XGBoost、不做深度學習），但架構必須預留介面（見 §2）。

使用情境：
- 單人、正面拍攝、室內光線充足；筆電放桌上，人退後約 2 公尺、全身入鏡
- 一回合 1 分鐘、20 個提示，每個提示限時 3 秒
- 手臂一律由身體側面舉起（側舉）、一次只舉一隻手；不做前舉（正面拍攝時 2D 角度會縮短）

### 11 個類別

| label | 說明 |
|---|---|
| `left_45` `left_90` `left_135` `left_180` | 左手側舉（受試者本人的左手） |
| `right_45` `right_90` `right_135` `right_180` | 右手側舉 |
| `squat` | 蹲下 |
| `jump` | 跳躍 |
| `none` | 無動作（站立、走動、晃動、抓頭、調整站姿、手臂自然擺動） |

肩角定義：上臂（肩→肘）與軀幹（肩→髖）的夾角。

### 驗收標準（雛型階段）
- 一回合 20 個提示，至少正確判定 17 個
- 姿勢到位後 ≤ 1.5 秒完成判定
- 輸出逐題結果 CSV 與視窗特徵 CSV

## 1. 環境與課程程式碼慣例

- Python：使用 `pip install mediapipe` 能成功的版本（3.10–3.12 最保險；3.13 + mediapipe 0.10.32 也已測過可安裝）
- 套件：見 `requirements.txt`（mediapipe、opencv-python、numpy、pandas、pillow）
- 模型檔：`models/pose_landmarker_full.task`（和課程 W2 同一個；FPS 不夠可換 `pose_landmarker_lite.task`）。
  沒有的話從 MediaPipe 官方下載：
  `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
- **沿用課程寫法**：
  - W2：MediaPipe **Tasks API**，`vision.PoseLandmarker`，`RunningMode.VIDEO`，`landmarker.detect_for_video(mp_image, timestamp_ms)`。不要改用舊版 `mp.solutions.pose`。
  - W4：三點夾角用向量內積 + `np.arccos(np.clip(...))`，長度乘積 ≤ 1e-12 時回傳 None。
  - W1/W3：路徑用 `pathlib`，CSV 用 `encoding="utf-8-sig"`、`newline=""`。

## 2. 架構原則（不可違反）

1. **規則法與之後的模型吃同一份「視窗特徵」**。規則法不得直接讀 landmarks，只能讀 `window.py` 產生的特徵 dict。
2. **分類器統一介面**：`classifiers/base.py`
   ```python
   class Classifier:
       def predict(self, window_features: dict) -> tuple[str, float]:  # (label, prob)
   ```
   規則法沒有機率，先固定回傳 1.0。之後的 `classifiers/ml.py` 實作同一介面，由 `config.CLASSIFIER = "rule" | "ml"` 切換。
3. **輸入來源可切換 Webcam / 影片檔**（`--video path`），方便重複測試，也是 Demo 備案。
4. **時間一律用 timestamp_ms**，視窗長度、間隔、延遲都用毫秒計算，不寫死幀數（筆電實際 FPS 可能不到 30）。
5. **所有門檻、參數放 `config.py`**，程式內不寫魔術數字。

### 目錄結構

```
stretch_game/
├── plan.md
├── requirements.txt
├── config.py          # ✅ 已完成（M2 之後持續加參數）
├── source.py          # ✅ 已完成：Webcam / 影片輸入，回傳 (frame, timestamp_ms)
├── pose.py            # ✅ 已完成：PoseLandmarker 包裝，回傳 (33,4) [x_px, y_px, z, visibility]
├── features.py        # ✅ 已完成：正規化、夾角、逐幀特徵
├── ui.py              # ✅ 已完成：骨架、角度標示、HUD（M4 擴充）
├── main.py            # ✅ 已完成：M1 除錯畫面（M2 起擴充成多模式）
├── calibration.py     # M2：站立校正
├── window.py          # M2：滑動視窗與統計
├── classifiers/
│   ├── __init__.py
│   ├── base.py        # M2
│   └── rule.py        # M2
├── judge.py           # M3：連續視窗確認
├── game.py            # M4：遊戲狀態機
├── logger.py          # M5：CSV 輸出
├── models/            # 放 .task 模型檔（不進版控）
├── outputs/           # 執行輸出
└── tests/
    ├── test_features.py   # ✅ 已完成，全部通過
    ├── test_window.py     # M2
    ├── test_rule.py       # M2
    ├── test_judge.py      # M3
    └── test_game.py       # M4
```

## 3. 已完成部分（M0、M1）— 先驗證再往下

開始前先執行 `python tests/test_features.py`，應全部通過。

已確定的設計（後續程式要遵守）：
- **座標**：`pose.py` 已把 MediaPipe 的正規化 x/y 乘回像素。16:9 畫面直接用正規化座標算夾角會失真。
- **正規化**：原點＝髖部中心、單位＝肩寬（像素）、y 往上為正、x 往**受試者本人左邊**為正（用左右肩相對位置決定方向，因此鏡像與否不影響特徵）。
- **偵測一律對未翻轉的原始影格做**；顯示時才鏡像。繪圖順序：骨架畫在原始影格 → 翻轉 → 文字最後畫。
- **visibility**：12 個關鍵點（肩、肘、腕、髖、膝、踝）任一 < `VISIBILITY_THRESHOLD` 時該幀無效，畫面提示 “Step back / full body”。
- 逐幀特徵 `FEATURE_NAMES`（12 個）：
  `L/R_shoulder_angle`、`L/R_elbow_angle`、`L/R_knee_angle`、`L/R_wrist_dx`、`L/R_wrist_dy`（手腕相對同側肩膀，dx 往身體外側為正）、`hip_ankle_height`、`torso_tilt`
- `raw`：`hip_y_px`、`shoulder_width_px`（給校正與跳躍/蹲下判斷用）

**已知限制（重要）**：髖部中心是正規化原點，所以正規化後「髖部高度」永遠是 0，跳躍/蹲下必須和「站立校正基準」比較（見 M2）。另外正面拍攝時蹲下的膝蓋是往鏡頭方向彎，**2D 膝角不一定明顯變小**，蹲下判斷要以髖部高度為主、膝角為輔。

## 4. 里程碑

### M2 — 站立校正、滑動視窗、規則分類

**calibration.py**
- 開局請使用者自然站立 `CALIBRATION_SEC = 3` 秒，收集有效幀
- 基準值取中位數：`base_hip_y_px`、`base_shoulder_width_px`、`base_hip_ankle_height`
- 有效幀少於期望幀數的 60% → 校正失敗，要求重來
- 校正後每幀加一個特徵：`hip_rise = (base_hip_y_px - hip_y_px) / base_shoulder_width_px`（往上為正）

**window.py**
- 以 timestamp 維護緩衝；每 `WINDOW_STRIDE_MS = 200` 產生一個視窗，長度 `WINDOW_MS`（可設 500 或 1000，預設 1000）
- 視窗內有效幀比例 < `MIN_VALID_RATIO = 0.7` → 該視窗無效（回傳 None，不送分類）
- 對 13 個逐幀特徵（12 + `hip_rise`）各算 `mean`、`std`、`max`、`min`、`delta`（末幀 − 首幀），
  欄位名格式 `{feature}_{stat}`，例如 `L_shoulder_angle_mean`，共 65 維；欄位順序固定，存成 `WINDOW_FEATURE_NAMES`

**classifiers/rule.py**（門檻全部放 config，以下是初始值，實測後再調）
判斷優先順序：jump → squat → 手臂 → none
1. `jump`：`hip_rise_max > JUMP_RISE = 0.25` 且 `hip_rise_max - hip_rise_min > JUMP_RANGE = 0.35`
2. `squat`：`hip_rise_mean < SQUAT_DROP = -0.5`（髖部比站立低半個肩寬以上）
3. 手臂（左右各自判斷）：
   - 舉起條件：`shoulder_angle_mean >= ARM_RAISED_MIN = 25` 且 `elbow_angle_mean >= ELBOW_STRAIGHT_MIN = 140` 且 `shoulder_angle_std <= ARM_STABLE_STD = 12`（維持住才算）
   - 兩手同時舉起 → `none`（遊戲規定一次一隻手）
   - 角度分級：取最接近者，分界 `67.5 / 112.5 / 157.5`
4. 其他 → `none`

**main.py 擴充**：`--mode viewer`（現有 M1 畫面）、`--mode classify`（加上校正流程，畫面顯示即時預測 label 與目前視窗統計）

**測試**：
- `test_window.py`：用合成時間序列驗證視窗切法、統計值、無效視窗、時間戳記非均勻（模擬 FPS 20）時視窗長度仍正確
- `test_rule.py`：用 `tests/test_features.py` 的 `make_pose()` 產生合成序列，驗證 11 類都能分對；兩手同舉 → none；手臂移動中（std 大）→ none

**需使用者實測**：`python main.py --mode classify`，11 類逐一做，確認預測正確；照 §5 的無動作腳本做一遍，確認不會被誤判。

### M3 — 判定機制

**judge.py**
- 連續 `CONFIRM_WINDOWS = 3` 個有效視窗預測同一類、且 `prob > PROB_THRESHOLD = 0.7` → 判定成立，回傳 `(label, confirm_ts)`
- 無效視窗會打斷連續計數
- 提供 `reset()`，換題時呼叫

**延遲量測**：`main.py --mode classify --log` 輸出每個視窗的 `(ts, pred, prob, confirmed)`；寫一個 `tools/latency.py` 讀 CSV，計算「單一視窗第一次預測為某類」到「該類判定成立」的時間。

**測試**：`test_judge.py`（連續計數、中斷、機率不足、reset）

**需使用者實測**：0.5 秒和 1 秒兩種視窗的判定延遲都要 ≤ 1.5 秒。

### M4 — 遊戲

**game.py 狀態機**
```
CALIBRATE → COUNTDOWN(3s) → PROMPT(3s) → FEEDBACK(1s) → WAIT_NEUTRAL → PROMPT ... → RESULT
```
- 20 個提示：10 個動作類別（不含 none）各 2 次，打亂順序，且不連續出現同一類
- PROMPT：
  - 只有判定為**目標類別**才算成功並結束該題（舉到 135° 途中經過 45°、90° 是正常的，不能因為先判到 90° 就判錯）
  - 期間判定成立的非目標類別記錄下來；逾時則該題失敗，`predicted` 記錄出現最多次的非目標類別（沒有則記 `none`）
- WAIT_NEUTRAL：等 judge 判到 `none`（最多 2 秒）再出下一題，避免上一題姿勢延續；換題時 `judge.reset()`
- `reaction_time` = 判定成立時間 − 提示出現時間

**畫面（ui.py 擴充）**
- 側舉提示：在對應肩膀畫目標角度參考線。起點＝肩膀，方向＝肩→髖單位向量往身體外側旋轉目標角度，長度＝上臂＋前臂長度（從當前 landmarks 算）。人側傾時參考線會跟著轉。
- 蹲下/跳躍提示：只顯示文字
- 顯示提示文字、倒數條、目前分數、題號；判定成功/失敗的回饋
- 中文顯示：cv2.putText 不支援中文，用 PIL 繪製。字型路徑放 config：Windows `C:/Windows/Fonts/msjh.ttc`、macOS `/System/Library/Fonts/PingFang.ttc`；找不到字型就退回英文
- 鏡像顯示時參考線與文字位置要正確（參考 `ui.draw_angle_labels` 的做法）

**RESULT 畫面**：總分、正確率、平均反應時間、各類別錯誤次數；按 r 再玩、q 離開

**入口**：`python main.py --mode game`

**測試**：`test_game.py` 用假的 judge 事件序列驅動狀態機（不需鏡頭），驗證：成功、逾時、途經其他角度不判錯、WAIT_NEUTRAL 逾時、提示序列平衡且不連續重複

### M5 — 輸出與自我驗收

**logger.py**，輸出到 `outputs/`，檔名帶時間：
- `game_results_YYYYmmdd_HHMMSS.csv`：`player, round_id, prompt_idx, target, result(correct/timeout), predicted, reaction_time_s, prompt_start_ms`
- `window_features_YYYYmmdd_HHMMSS.csv`：`timestamp_ms, target(遊戲中的目標，非遊戲時空白), pred, prob, confirmed, *WINDOW_FEATURE_NAMES`
  — **這份格式就是之後機器學習的訓練資料格式**，欄位順序必須和 `WINDOW_FEATURE_NAMES` 一致
- 遊戲開始前可輸入玩家代號（`--player A`）

**需使用者實測**：5 位組員各玩一回合，記錄是否 ≥ 17/20、判定 ≤ 1.5 秒；所有失敗案例整理成「規則法失效案例」（之後說明為何需要機器學習）。

## 5. 「無動作」測試腳本（每人 1 分鐘）

| 時間 | 動作 | 容易被誤判成 |
|---|---|---|
| 0–10 秒 | 自然站立 | 基準 |
| 10–20 秒 | 原地走動 | 跳躍 |
| 20–30 秒 | 身體小幅晃動 | 蹲下、跳躍 |
| 30–40 秒 | 抓頭、整理頭髮 | 135°、180° 側舉 |
| 40–50 秒 | 調整站姿 | 蹲下 |
| 50–60 秒 | 手臂自然擺動 | 45° 側舉 |

## 6. 給 Claude Code 的工作規則

- **GitHub Repo**：`https://github.com/wwwkpo/shoulder_movement_game.git`（固定，每次 push 都推到這裡）
- 依 M2 → M5 順序；每個里程碑完成後：跑 `tests/` 全部測試 → 列出改了哪些檔案 → 列出需要使用者 Webcam 實測的項目 → **停下來等回報**
- 沒有鏡頭可用時，以合成資料單元測試 + `--video` 模式驗證；不要假設實測結果
- 不更動 `features.py` 的座標約定與 `FEATURE_NAMES` 既有順序（可以往後新增）
- 新參數一律加到 `config.py` 並附中文註解
- 程式註解用繁體中文，畫面文字用中文（PIL），除錯畫面可用英文
- 不實作機器學習；只確保 `classifiers/base.py` 介面與 `window_features` CSV 格式能讓之後直接接上

## 7. 之後（不在本計畫範圍）

- 錄製工具：沿用 M4 的提示與參考線畫面，加上自動記錄類別與起訖時間
- `classifiers/ml.py`：讀 `window_features` CSV 訓練 Logistic Regression 與 Random Forest（不做 XGBoost），留一人交叉驗證，實作同一個 `predict()` 介面
