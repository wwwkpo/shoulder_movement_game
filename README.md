# 課間肩部伸展體感遊戲

課程「影像辨識應用及動作辨識導論」期中專題。
用筆電 Webcam + MediaPipe Pose 辨識手臂側舉角度與全身動作，做成「依提示做出指定伸展姿勢」的體感遊戲。

## 環境需求

- Python 3.10 ~ 3.12（或 3.13，已測可裝 mediapipe 0.10.32）
- 套件：`pip install -r requirements.txt`
- 模型檔：`models/pose_landmarker_full.task`（需自行下載，見下方說明）

### 下載模型檔

```bash
mkdir models
curl -o models/pose_landmarker_full.task \
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
```

Windows（PowerShell）：

```powershell
mkdir models
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task" -OutFile "models/pose_landmarker_full.task"
```

## 使用方式

### M1 — 除錯畫面（骨架 + 角度）

```bash
python main.py                    # Webcam
python main.py --video clip.mp4   # 影片檔
python main.py --log              # 同時輸出逐幀特徵 CSV
```

### M2 — 即時分類（規則法）

```bash
python main.py --mode classify
python main.py --mode classify --video clip.mp4
python main.py --mode classify --log   # 輸出視窗特徵 CSV（機器學習訓練資料格式）
```

#### 操作按鍵

| 按鍵 | 功能 |
|------|------|
| `q` / `Esc` | 離開 |
| `m` | 切換鏡像顯示 |
| `Space` | 暫停 / 繼續 |
| `c` | 重新校正（classify 模式） |

## 專案結構

```
stretch_game/
├── config.py          # 所有參數（門檻、視窗長度…）集中管理
├── source.py          # Webcam / 影片輸入
├── pose.py            # MediaPipe PoseLandmarker 包裝
├── features.py        # 逐幀特徵計算（正規化、夾角）
├── calibration.py     # 站立校正
├── window.py          # 滑動視窗統計（65 維視窗特徵）
├── classifiers/
│   ├── base.py        # 分類器統一介面
│   └── rule.py        # 規則法分類器（含 explain()）
├── ui.py              # 骨架、角度標示、HUD 繪製
├── main.py            # 主入口（--mode viewer / classify）
├── models/            # 放 .task 模型檔（不進版控）
├── outputs/           # 執行輸出 CSV（不進版控）
└── tests/
    ├── test_features.py
    ├── test_window.py
    └── test_rule.py
```

## 11 個辨識類別

| 類別 | 說明 |
|------|------|
| `left_45` / `left_90` / `left_135` / `left_180` | 左手側舉（受試者本人的左手） |
| `right_45` / `right_90` / `right_135` / `right_180` | 右手側舉 |
| `squat` | 蹲下 |
| `jump` | 跳躍 |
| `none` | 無動作（站立、走動、晃動…） |

## 執行測試

```bash
python -X utf8 tests/test_features.py
python -X utf8 tests/test_window.py
python -X utf8 tests/test_rule.py
```

## 開發進度

| 里程碑 | 內容 | 狀態 |
|--------|------|------|
| M0 | 環境、MediaPipe、基本骨架顯示 | ✅ 完成 |
| M1 | 逐幀特徵計算、除錯畫面 | ✅ 完成 |
| M2 | 站立校正、滑動視窗、規則分類 | ✅ 完成 |
| M3 | 連續視窗判定機制 | 待開發 |
| M4 | 遊戲狀態機、UI | 待開發 |
| M5 | CSV 輸出、自我驗收 | 待開發 |

## 授課課程

影像辨識應用及動作辨識導論
