# 基於明星比例的智能瘦臉系統

## 功能說明

這個系統可以：
1. 分析明星照片的臉部比例（臉寬、臉長、下巴寬度等）
2. 即時比較您的臉部比例與明星的差異
3. 自動調整變換參數，讓您的臉接近明星的理想比例
4. 提供視覺化的比例對比

## 使用步驟

### 1. 準備明星照片

找一張您喜歡的明星正面照，要求：
- 清晰的正面照
- 臉部完整可見
- 光線均勻
- 建議尺寸：至少 500x500 像素

推薦明星範例（以臉部輪廓聞名）：
- 女性：劉亦菲、Angelababy、高圓圓、朴信惠
- 男性：金秀賢、胡歌、彭于晏

將照片保存到：`.assets/images/celebrity.jpg`

### 2. 分析明星照片

執行分析腳本來提取臉部比例：

```powershell
python analyze_celebrity.py .assets/images/celebrity.jpg
```

這會：
- 顯示標註了關鍵點的圖片
- 輸出臉部比例數據
- 保存 `celebrity_analyzed.jpg`（標註版）
- 保存 `celebrity.json`（比例數據）

### 3. 運行即時變換

```powershell
python main.py
```

系統會：
- 自動載入明星的臉部比例
- 即時分析您的臉部比例
- 根據差異動態調整瘦臉強度
- 在畫面上顯示比例對比

### 4. 畫面資訊說明

運行時，視窗上會顯示：
- **Target**: 明星的臉寬/臉長比例（目標值）
- **Your**: 您當前的臉寬/臉長比例
- **Strength**: 當前的變換強度（自動調整）
- **FPS**: 每秒幀數
- **彩色點**：
  - 綠點：鼻尖（變換中心）
  - 藍點：左臉頰
  - 紅點：右臉頰

## 比例說明

### 臉寬/臉長比例 (face_ratio)
- **0.6-0.7**：標準瓜子臉（東亞審美理想）
- **0.7-0.8**：中等臉型
- **0.8-0.9**：較寬臉型

### 下巴/臉寬比例 (jaw_ratio)
- **0.6-0.7**：尖下巴（V臉）
- **0.7-0.8**：標準下巴
- **0.8-0.9**：寬下巴

## 進階使用

### 手動調整參數

如果您想要更細緻的控制，可以修改 `main.py` 中的參數：

```python
# 在 main.py 中找到這些行並調整
default_strength = 0.35      # 預設強度 (0.1-0.7)
default_radius_ratio = 0.15  # 預設半徑比例 (0.1-0.3)
```

### 分析多個明星

您可以分析多個明星並保存不同的配置：

```powershell
# 分析明星 A
python analyze_celebrity.py .assets/images/celebrity_a.jpg

# 分析明星 B
python analyze_celebrity.py .assets/images/celebrity_b.jpg
```

然後在 `main.py` 中修改要使用的照片：

```python
celebrity_image = ".assets/images/celebrity_a.jpg"  # 或 celebrity_b.jpg
```

## 常見問題

### Q: 為什麼檢測不到我的臉？
A: 確保：
- 攝像頭正常工作
- 光線充足
- 正面對準攝像頭
- 臉部沒有被遮擋

### Q: 為什麼效果不明顯？
A: 可能原因：
- 您的臉部比例已經很接近明星了！
- 可以手動增加 `strength` 參數
- 確保明星照片正確載入

### Q: 可以保存變換後的視頻嗎？
A: 可以，在 `main.py` 中添加 VideoWriter：

```python
# 在主循環開始前
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (w, h))

# 在主循環中
out_video.write(out[:, :, ::-1])

# 結束時
out_video.release()
```

## 技術細節

### 使用的關鍵點（MediaPipe 468 landmarks）

- **234**: 左臉頰外側
- **454**: 右臉頰外側
- **10**: 額頭頂部
- **152**: 下巴底部
- **172**: 左下顎
- **397**: 右下顎
- **1**: 鼻尖

### 變換算法

使用局部網格變形（Local Mesh Warping）：
1. 以臉頰為中心建立變形區域
2. 計算每個像素到中心的距離
3. 根據距離應用衰減函數（falloff）
4. 向目標方向（鼻子）移動像素
5. 使用 `cv2.remap` 進行重映射

### 自適應參數計算

```python
strength = 0.3 + min(jaw_diff * 2.0, 0.4)
radius_ratio = 0.15 + min(face_ratio_diff * 0.3, 0.1)
```

- `jaw_diff`: 您與明星的下巴比例差異
- `face_ratio_diff`: 您與明星的臉型比例差異
- 系統自動根據差異調整強度

## 按鍵操作

- **Q**: 退出程式

## 系統要求

- Python 3.11+
- MediaPipe
- OpenCV
- 攝像頭

## 未來改進方向

- [ ] 添加更多變換選項（大眼、高鼻樑等）
- [ ] 支援視頻文件輸入
- [ ] 實時參數調整（滑桿）
- [ ] 多人臉支援
- [ ] 保存個人配置文件
- [ ] 添加濾鏡效果

## 免責聲明

此工具僅供娛樂和學習使用。請理性看待外貌，每個人都有獨特的美。過度使用美顏工具可能導致對自我形象的扭曲認知。
