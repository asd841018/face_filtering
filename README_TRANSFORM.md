# 明星臉部比例變換系統

這個系統可以分析明星的臉部比例，然後將你的臉部按照明星的比例進行變換。使用 **Delaunay 三角變換技術**，效果更自然、更專業。

## 📋 使用流程

### 第一步：分析明星照片

首先，找一張你喜歡的明星照片（正面、清晰），然後分析其臉部比例：

```powershell
python analyze_celebrity.py .assets\images\celebrity_photo.jpg
```

這會生成：
- `celebrity_photo_analyzed.jpg` - 標註了關鍵特徵點的圖片
- `celebrity_photo.json` - 臉部比例數據

### 第二步：變換你的照片

使用明星的臉部比例來變換你的照片：

```powershell
# 基本用法（預設強度 0.5）
python face_transform.py my_photo.jpg celebrity_photo.json

# 指定變換強度（0~1）
python face_transform.py my_photo.jpg celebrity_photo.json 0.7

# 指定輸出檔案
python face_transform.py my_photo.jpg celebrity_photo.json 0.6 my_result.jpg
```

## 🎯 參數說明

### 變換強度（intensity）

- `0.0` - 完全不變
- `0.3` - 輕微調整（自然）
- `0.5` - 適度調整（推薦）✨
- `0.7` - 明顯調整
- `1.0` - 完全變成目標比例

## 🔧 技術特點

### 1. Delaunay 三角變換
- 使用 Delaunay 三角剖分技術
- 將臉部分割成多個三角形
- 對每個三角形進行獨立的仿射變換
- 效果比簡單的 landmark 位移更自然

### 2. 關鍵比例調整
- **臉寬/臉長比例**：控制臉型（長臉 vs 圓臉）
- **下巴/臉寬比例**：控制下巴尖度

### 3. 智能調整
- 漸進式調整（上半臉變化小，下半臉變化大）
- 保持臉部中心線穩定
- 邊界點保護，避免圖片變形

## 📊 比較：舊方法 vs 新方法

### 舊方法（main.py）
❌ 只使用 landmark 點位移
❌ 簡單的局部變形（local_shrink）
❌ 可能產生不自然的扭曲
❌ 無法基於明星比例調整

### 新方法（face_transform.py）
✅ 使用 Delaunay 三角變換
✅ 基於明星臉部比例科學調整
✅ 變換更自然、更連續
✅ 可調整變換強度

## 🌟 推薦明星參考

不同臉型的明星推薦：

**女性：**
- 瓜子臉：劉亦菲、Angelababy
- 鵝蛋臉：劉詩詩、高圓圓
- 精緻小臉：楊冪、迪麗熱巴

**男性：**
- 立體五官：吳彥祖、彭于晏
- 斯文帥氣：王力宏、金城武
- 硬漢風格：張震、胡歌

## ⚠️ 注意事項

1. **照片要求**
   - 正面照、光線充足
   - 臉部完整可見
   - 解析度不要太低

2. **明星照片選擇**
   - 最好是高清正面照
   - 無過度修圖
   - 表情自然

3. **變換強度建議**
   - 第一次嘗試用 0.3-0.5
   - 太高可能失真
   - 可以多次嘗試不同強度

## 💡 示例

```powershell
# 1. 分析劉亦菲的照片
python analyze_celebrity.py .assets\images\liuyifei.jpg

# 2. 用她的比例變換自己的照片（溫和）
python face_transform.py my_selfie.jpg liuyifei.json 0.4 result_gentle.jpg

# 3. 用她的比例變換自己的照片（明顯）
python face_transform.py my_selfie.jpg liuyifei.json 0.7 result_obvious.jpg
```

## 🔍 技術細節

使用了約 **80+ 個關鍵特徵點**：
- 臉部輪廓：~30 個點
- 眼睛：~16 個點
- 鼻子：~7 個點
- 嘴巴：~20 個點
- 眉毛：~10 個點
- 邊界保護點：8 個點

每個三角形獨立計算仿射變換矩陣，確保平滑過渡。
