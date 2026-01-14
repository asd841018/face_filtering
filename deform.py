# STEP 1: Import the necessary modules.
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.interpolate import Rbf


def detect_skin_mask(img, face_landmarks, w, h):
    """
    檢測皮膚區域，返回皮膚遮罩
    使用 YCrCb 顏色空間進行皮膚檢測
    排除眼睛、眉毛、嘴巴等區域
    """
    # 轉換到 YCrCb 顏色空間（對皮膚檢測更有效）
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    
    # 皮膚顏色範圍（YCrCb 空間）
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # 創建皮膚遮罩
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # 形態學操作去除噪點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # 創建臉部區域遮罩（排除眼睛、眉毛、嘴巴）
    face_mask = np.zeros((h, w), dtype=np.uint8)
    
    # 獲取臉部輪廓點（橢圓形臉部區域）
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    face_points = []
    for idx in face_oval:
        x, y = to_pixel(face_landmarks[idx], w, h)
        face_points.append([x, y])
    
    face_points = np.array(face_points, dtype=np.int32)
    cv2.fillConvexPoly(face_mask, face_points, 255)
    
    # 排除眼睛區域
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    for eye_indices in [left_eye, right_eye]:
        eye_points = []
        for idx in eye_indices:
            x, y = to_pixel(face_landmarks[idx], w, h)
            eye_points.append([x, y])
        eye_points = np.array(eye_points, dtype=np.int32)
        cv2.fillConvexPoly(face_mask, eye_points, 0)
    
    # 排除嘴巴內部區域
    mouth_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    mouth_points = []
    for idx in mouth_inner:
        x, y = to_pixel(face_landmarks[idx], w, h)
        mouth_points.append([x, y])
    mouth_points = np.array(mouth_points, dtype=np.int32)
    cv2.fillConvexPoly(face_mask, mouth_points, 0)
    
    # 排除眉毛區域（可選，如果想保留眉毛質感）
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    
    for brow_indices in [left_eyebrow, right_eyebrow]:
        brow_points = []
        for idx in brow_indices:
            x, y = to_pixel(face_landmarks[idx], w, h)
            brow_points.append([x, y])
        brow_points = np.array(brow_points, dtype=np.int32)
        # 擴大眉毛區域
        hull = cv2.convexHull(brow_points)
        cv2.fillConvexPoly(face_mask, hull, 0)
    
    # 結合皮膚顏色遮罩和臉部區域遮罩
    final_mask = cv2.bitwise_and(skin_mask, face_mask)
    
    # 高斯模糊遮罩邊緣，使過渡更自然
    final_mask = cv2.GaussianBlur(final_mask, (7, 7), 0)
    
    return final_mask


def smooth_skin(img, mask, smooth_level=3, detail_preserve=0.7):
    """
    對皮膚區域進行平滑處理
    
    參數：
    - img: 輸入圖像
    - mask: 皮膚遮罩（0-255）
    - smooth_level: 平滑程度（1-5，越大越平滑）
    - detail_preserve: 細節保留程度（0-1，越大保留越多細節）
    """
    if smooth_level == 0:
        return img
    
    # 根據 smooth_level 調整參數
    d = smooth_level * 3 + 6  # bilateral filter 的直徑
    sigma_color = 40 + smooth_level * 20  # 顏色空間標準差
    sigma_space = 40 + smooth_level * 20  # 座標空間標準差
    
    # 方法1：雙邊濾波（保留邊緣的同時平滑）
    smoothed = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    
    # 方法2：結合高斯模糊和原圖（可選，用於更強的平滑）
    if smooth_level >= 3:
        kernel_size = smooth_level * 2 + 1
        gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        # 混合雙邊濾波和高斯模糊
        smoothed = cv2.addWeighted(smoothed, 0.7, gaussian, 0.3, 0)
    
    # 高頻細節提取（用於保留一些皮膚質感）
    if detail_preserve > 0:
        # 使用高斯模糊提取低頻部分
        low_freq = cv2.GaussianBlur(img, (15, 15), 0)
        # 高頻細節 = 原圖 - 低頻
        high_freq = cv2.subtract(img, low_freq)
        # 將部分高頻細節加回平滑後的圖像
        smoothed = cv2.add(smoothed, (high_freq * detail_preserve).astype(np.uint8))
    
    # 將遮罩歸一化到 0-1
    mask_norm = mask.astype(np.float32) / 255.0
    mask_norm = np.expand_dims(mask_norm, axis=2)  # 擴展到 3 通道
    
    # 混合原圖和平滑後的圖像
    result = (img * (1 - mask_norm) + smoothed * mask_norm).astype(np.uint8)
    
    return result


def get_face_mesh_triangles():
    """
    返回 MediaPipe Face Mesh 的三角網格拓撲結構
    這些三角形索引定義了臉部的3D網格結構
    """
    # MediaPipe 468 landmarks 的部分三角網格連接
    # 這裡包含主要的臉部區域三角形
    triangles = [
        # 臉頰區域
        [234, 93, 132], [93, 132, 58], [132, 58, 172], [58, 172, 136],
        [172, 136, 150], [136, 150, 149], [150, 149, 176], [149, 176, 148],
        [176, 148, 152], [148, 152, 377], [152, 377, 400],
        
        # 右臉頰區域
        [454, 323, 361], [323, 361, 288], [361, 288, 397], [288, 397, 365],
        [397, 365, 379], [365, 379, 378], [379, 378, 400], [378, 400, 377],
        [400, 377, 152],
        
        # 額頭區域
        [10, 338, 297], [10, 297, 338], [297, 332, 284], [332, 284, 251],
        [284, 251, 389], [251, 389, 356], [389, 356, 454], [356, 454, 323],
        [10, 109, 67], [109, 67, 103], [67, 103, 54], [103, 54, 21],
        [54, 21, 162], [21, 162, 127], [162, 127, 234], [127, 234, 93],
        
        # 鼻子區域
        [4, 5, 195], [5, 195, 197], [195, 197, 6], [197, 6, 168],
        [6, 168, 1], [168, 1, 197], [1, 197, 195], [1, 195, 5],
        
        # 嘴巴周圍
        [61, 146, 91], [146, 91, 181], [91, 181, 84], [181, 84, 17],
        [84, 17, 314], [17, 314, 405], [314, 405, 321], [405, 321, 375],
        [321, 375, 291], [375, 291, 61],
        
        # 下巴區域
        [152, 148, 176], [152, 377, 400], [176, 149, 150],
        [400, 378, 379], [148, 176, 149], [377, 400, 378],
    ]
    
    return triangles


def get_face_control_points(face_landmarks, w, h):
    """
    1. 獲取用於變形的臉部控制點（精簡版，減少控制點提升速度）
    返回關鍵點索引和像素座標
    """
    # 定義臉部各區域的關鍵點（減少數量以提升速度）
    # 左臉頰（精簡）
    LEFT_CHEEK = [234, 132, 172, 150, 176]
    # 右臉頰（精簡）
    RIGHT_CHEEK = [454, 361, 397, 379, 400]
    # 下巴（精簡）
    CHIN = [152, 176, 149, 377, 400]
    # 鼻子（固定不動）
    NOSE = [1, 4, 168]
    # 眼睛周圍（固定不動，精簡）
    LEFT_EYE = [33, 133, 145, 153]
    RIGHT_EYE = [362, 263, 374, 380]
    # 嘴巴周圍（精簡）
    MOUTH = [61, 91, 17, 314, 321, 291]
    
    # 組合所有控制點
    all_indices = (LEFT_CHEEK + RIGHT_CHEEK + CHIN + 
                   NOSE + LEFT_EYE + RIGHT_EYE + MOUTH)
    
    # 去除重複
    all_indices = list(set(all_indices))
    all_indices.sort()
    
    src_points = []
    for idx in all_indices:
        x, y = to_pixel(face_landmarks[idx], w, h)
        src_points.append([x, y])
    
    return np.array(src_points, dtype=np.float32), all_indices, LEFT_CHEEK, RIGHT_CHEEK, CHIN


def apply_mesh_deformation(src_points, indices, left_cheek_indices, right_cheek_indices, 
                          chin_indices, face_landmarks, w, h, 
                          cheek_strength=0.2, chin_strength=0.15):
    """
    2. 應用基於網格的變形
    - cheek_strength: 臉頰往內偏移強度 (0~1)
    - chin_strength: 下巴往上收縮強度 (0~1)
    """
    dst_points = src_points.copy()
    
    # 取得鼻尖作為中心參考點
    nose_x, nose_y = to_pixel(face_landmarks[1], w, h)
    
    # 取得下巴最低點作為參考
    chin_bottom_y = max([to_pixel(face_landmarks[152], w, h)[1],
                         to_pixel(face_landmarks[152], w, h)[1]])
    
    # 對左臉頰點進行偏移（往鼻子方向）
    for idx in left_cheek_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            # 計算往鼻子方向的向量
            dx = nose_x - x
            dy = nose_y - y
            # 應用偏移（根據距離鼻子的遠近調整強度）
            distance_factor = np.sqrt(dx**2 + dy**2) / w  # 正規化距離
            strength = cheek_strength * (0.5 + 0.5 * distance_factor)  # 距離越遠效果越明顯
            dst_points[orig_idx][0] = x + dx * strength
            dst_points[orig_idx][1] = y + dy * strength * 0.5  # Y方向偏移較小
    
    # 對右臉頰點進行偏移（往鼻子方向）
    for idx in right_cheek_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            dx = nose_x - x
            dy = nose_y - y
            distance_factor = np.sqrt(dx**2 + dy**2) / w
            strength = cheek_strength * (0.5 + 0.5 * distance_factor)
            dst_points[orig_idx][0] = x + dx * strength
            dst_points[orig_idx][1] = y + dy * strength * 0.5
    
    # 對下巴點進行往上收縮
    for idx in chin_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            # 往上和往中心偏移
            dy = chin_bottom_y - y
            dx = nose_x - x
            # 根據距離下巴最低點的遠近調整強度
            distance_factor = abs(dy) / h
            strength = chin_strength * distance_factor
            dst_points[orig_idx][0] = x + dx * strength * 0.3
            dst_points[orig_idx][1] = y - abs(dy) * strength
    
    return dst_points


def mesh_warp_image(img, src_points, dst_points):
    """
    3. 使用仿射變換對圖片進行基於網格的變形
    將圖片劃分為多個三角形網格，對每個三角形進行仿射變換
    """
    h, w = img.shape[:2]
    
    # 創建輸出圖片
    output = np.zeros_like(img)
    
    # 計算 Delaunay 三角剖分
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    
    # 添加目標點到三角剖分
    for point in dst_points:
        try:
            subdiv.insert((float(point[0]), float(point[1])))
        except:
            continue
    
    # 獲取三角形列表
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    # 對每個三角形進行變形
    for t in triangles:
        # 三角形頂點座標
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        # 檢查三角形是否在圖片範圍內
        if (pt1[0] < 0 or pt1[0] >= w or pt1[1] < 0 or pt1[1] >= h or
            pt2[0] < 0 or pt2[0] >= w or pt2[1] < 0 or pt2[1] >= h or
            pt3[0] < 0 or pt3[0] >= w or pt3[1] < 0 or pt3[1] >= h):
            continue
        
        # 找到目標三角形對應的源三角形
        dst_tri = np.float32([pt1, pt2, pt3])
        
        # 在源點中找到最接近的點
        src_tri = []
        for dst_pt in dst_tri:
            distances = np.sqrt(np.sum((dst_points - dst_pt)**2, axis=1))
            min_idx = np.argmin(distances)
            if distances[min_idx] < 5:  # 只有距離很近的點才認為是對應點
                src_tri.append(src_points[min_idx])
            else:
                src_tri.append(dst_pt)  # 如果找不到對應點，使用原座標
        
        src_tri = np.float32(src_tri)
        
        # 計算仿射變換矩陣
        try:
            warp_mat = cv2.getAffineTransform(src_tri, dst_tri)
        except:
            continue
        
        # 獲取三角形的邊界框
        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)
        
        # 裁剪三角形區域
        src_tri_cropped = src_tri - [r1[0], r1[1]]
        dst_tri_cropped = dst_tri - [r2[0], r2[1]]
        
        # 裁剪源圖片
        src_crop = img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
        
        if src_crop.shape[0] == 0 or src_crop.shape[1] == 0:
            continue
        
        # 應用仿射變換
        try:
            dst_crop = cv2.warpAffine(src_crop, warp_mat, (r2[2], r2[3]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
        except:
            continue
        
        # 創建三角形遮罩
        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_cropped), (1.0, 1.0, 1.0), 16, 0)
        
        # 混合到輸出圖片
        try:
            output[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = \
                output[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * (1 - mask) + dst_crop * mask
        except:
            continue
    
    return output.astype(np.uint8)


def simple_mesh_warp_image(img, src_points, dst_points, grid_resolution=40):
    """
    4. 使用 RBF (Radial Basis Function) 進行快速網格變形
    使用降採樣網格加速計算，然後插值放大到原始解析度
    grid_resolution: 網格解析度（越小越快）
    """
    h, w = img.shape[:2]
    
    # 添加邊界控制點以固定圖片邊緣
    margin = max(40, min(w, h) // 10)
    border_points = []
    
    # 四個角
    border_points.extend([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    
    # 邊界點
    for x in range(margin, w-margin, margin):
        border_points.extend([[x, 0], [x, h-1]])
    for y in range(margin, h-margin, margin):
        border_points.extend([[0, y], [w-1, y]])
    
    border_points = np.array(border_points, dtype=np.float32)
    
    # 合併臉部控制點和邊界點
    all_src_points = np.vstack([src_points, border_points])
    all_dst_points = np.vstack([dst_points, border_points])
    
    # 建立 RBF 插值（thin_plate 函數）
    rbf_x = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 0],
                function='thin_plate', smooth=0)
    rbf_y = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 1],
                function='thin_plate', smooth=0)
    
    # 使用降採樣網格加速計算
    grid_w = w // grid_resolution
    grid_h = h // grid_resolution
    
    grid_x_small = np.linspace(0, w-1, grid_w)
    grid_y_small = np.linspace(0, h-1, grid_h)
    grid_x_small, grid_y_small = np.meshgrid(grid_x_small, grid_y_small)
    
    # 在低解析度網格上計算變形
    map_x_small = rbf_x(grid_x_small, grid_y_small).astype(np.float32)
    map_y_small = rbf_y(grid_x_small, grid_y_small).astype(np.float32)
    
    # 插值放大到原始解析度
    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 執行重映射
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    
    return warped


def calculate_face_golden_ratios(face_landmarks, w, h):
    """
    計算臉部黃金比例參數
    返回當前比例和建議調整強度
    
    黃金比例標準：
    1. 臉寬/臉長 ≈ 0.618 (黃金比例倒數)
    2. 三庭比例：髮際線到眉毛 = 眉毛到鼻底 = 鼻底到下巴
    3. 五眼比例：臉寬 ≈ 5 個眼睛寬度
    4. 眼間距/臉寬 ≈ 0.46
    5. 鼻寬/臉寬 ≈ 0.25
    """
    GOLDEN_RATIO = 1.618
    
    # 關鍵點位置
    # 臉部輪廓
    left_face = to_pixel(face_landmarks[234], w, h)    # 左臉頰最外側
    right_face = to_pixel(face_landmarks[454], w, h)   # 右臉頰最外側
    top_face = to_pixel(face_landmarks[10], w, h)      # 額頭頂部
    chin = to_pixel(face_landmarks[152], w, h)         # 下巴底部
    
    # 眼睛位置
    left_eye_inner = to_pixel(face_landmarks[133], w, h)   # 左眼內角
    left_eye_outer = to_pixel(face_landmarks[33], w, h)    # 左眼外角
    right_eye_inner = to_pixel(face_landmarks[362], w, h)  # 右眼內角
    right_eye_outer = to_pixel(face_landmarks[263], w, h)  # 右眼外角
    
    # 鼻子
    nose_top = to_pixel(face_landmarks[168], w, h)     # 鼻樑
    nose_bottom = to_pixel(face_landmarks[2], w, h)    # 鼻尖
    nose_left = to_pixel(face_landmarks[98], w, h)     # 鼻翼左
    nose_right = to_pixel(face_landmarks[327], w, h)   # 鼻翼右
    
    # 眉毛
    left_eyebrow = to_pixel(face_landmarks[70], w, h)  # 左眉毛
    right_eyebrow = to_pixel(face_landmarks[300], w, h) # 右眉毛
    
    # 1. 計算臉寬和臉長
    face_width = abs(right_face[0] - left_face[0])
    face_height = abs(chin[1] - top_face[1])
    
    # 2. 臉寬/臉長比例（理想值 ≈ 0.618）
    width_height_ratio = face_width / face_height if face_height > 0 else 0
    ideal_width_height = 1.0 / GOLDEN_RATIO  # ≈ 0.618
    
    # 3. 眼睛寬度
    left_eye_width = abs(left_eye_outer[0] - left_eye_inner[0])
    right_eye_width = abs(right_eye_outer[0] - right_eye_inner[0])
    avg_eye_width = (left_eye_width + right_eye_width) / 2
    
    # 4. 眼間距
    eye_distance = abs(right_eye_inner[0] - left_eye_inner[0])
    eye_distance_ratio = eye_distance / face_width if face_width > 0 else 0
    ideal_eye_distance_ratio = 0.46
    
    # 5. 鼻寬
    nose_width = abs(nose_right[0] - nose_left[0])
    nose_width_ratio = nose_width / face_width if face_width > 0 else 0
    ideal_nose_width_ratio = 0.25
    
    # 6. 三庭比例
    # 上庭：髮際線到眉毛
    upper_third = abs(left_eyebrow[1] - top_face[1])
    # 中庭：眉毛到鼻底
    middle_third = abs(nose_bottom[1] - left_eyebrow[1])
    # 下庭：鼻底到下巴
    lower_third = abs(chin[1] - nose_bottom[1])
    
    # 計算建議的調整強度
    # 如果臉太寬 -> 增加臉頰收縮
    cheek_adjustment = 0.0
    if width_height_ratio > ideal_width_height:
        # 臉太寬，需要收縮
        cheek_adjustment = min(0.3, (width_height_ratio - ideal_width_height) * 0.8)
    
    # 如果下庭太長 -> 增加下巴收縮
    chin_adjustment = 0.0
    avg_third = (upper_third + middle_third + lower_third) / 3
    if lower_third > avg_third * 1.15:
        # 下巴太長
        chin_adjustment = min(0.25, (lower_third / avg_third - 1.0) * 0.5)
    
    ratios_info = {
        'face_width': face_width,
        'face_height': face_height,
        'width_height_ratio': width_height_ratio,
        'ideal_width_height': ideal_width_height,
        'eye_distance_ratio': eye_distance_ratio,
        'nose_width_ratio': nose_width_ratio,
        'upper_third': upper_third,
        'middle_third': middle_third,
        'lower_third': lower_third,
        'cheek_adjustment': cheek_adjustment,
        'chin_adjustment': chin_adjustment
    }
    
    return ratios_info


def apply_golden_ratio_deformation(src_points, indices, left_cheek_indices, right_cheek_indices,
                                   chin_indices, face_landmarks, w, h, ratios_info, strength=1.0):
    """
    根據黃金比例自動調整臉部
    strength: 整體調整強度 (0~1)，0為不調整，1為完全按黃金比例調整
    """
    # 使用黃金比例計算出的建議值
    cheek_strength = ratios_info['cheek_adjustment'] * strength
    chin_strength = ratios_info['chin_adjustment'] * strength
    
    # 調用原本的變形函數
    return apply_mesh_deformation(
        src_points, indices, left_cheek_indices, right_cheek_indices,
        chin_indices, face_landmarks, w, h, cheek_strength, chin_strength
    )


def to_pixel(lm, w, h):
    """將歸一化座標轉換為像素座標"""
    return int(lm.x * w), int(lm.y * h)


if __name__ == "__main__":
    # STEP 2: Create an FaceLandmarker object.
    model_path = ".assets/models/face_landmarker_v2_with_blendshapes.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # Params for mesh-based face reshaping
    cheek_strength = 0.10  # 臉頰往內偏移強度 (0~1)，越大臉頰越瘦
    chin_strength = 0.05   # 下巴往上收縮強度 (0~1)，越大下巴越尖
    use_simple_warp = True  # True: 使用快速 RBF, False: 使用三角網格變形
    grid_resolution = 50   # 網格解析度（20-50，越小越快但精度降低）
    
    # Golden Ratio mode
    use_golden_ratio = False  # 是否使用黃金比例自動調整
    golden_ratio_strength = 0.8  # 黃金比例調整強度 (0~1)
    show_ratios = False  # 是否顯示比例資訊
    
    # Skin smoothing
    enable_skin_smooth = True  # 是否啟用皮膚光滑
    smooth_level = 2  # 平滑程度（0-5，0為關閉）
    detail_preserve = 0.5  # 細節保留程度（0-1）
    
    # Display settings
    show_dual_view = True  # 是否顯示雙視窗（原圖+處理後）
    show_landmarks = False  # 是否顯示關鍵點
    landmark_mode = 0  # 關鍵點顯示模式：0=關閉, 1=所有468點, 2=控制點, 3=變形對比
    
    # STEP 3: Load the input image.
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("D:/work/face_filtering/.assets/images/Live_demo.mp4")
    
    print("=== Face Beauty Filter: Deform + Skin Smooth ===")
    print(f"Cheek strength: {cheek_strength}")
    print(f"Chin strength: {chin_strength}")
    print(f"Warp method: {'Fast RBF' if use_simple_warp else 'Triangle Mesh'}")
    print(f"Grid resolution: {grid_resolution}")
    print(f"Golden Ratio Mode: {'ON' if use_golden_ratio else 'OFF'}")
    print(f"Skin Smooth: {'ON' if enable_skin_smooth else 'OFF'} (Level: {smooth_level})")
    print(f"Dual View: {'ON' if show_dual_view else 'OFF'}")
    print(f"Landmarks: {'ON' if show_landmarks else 'OFF'} (Mode: {landmark_mode})")
    print("\n=== Controls ===")
    print("'q' - Quit")
    print("'d' - Toggle dual view (雙視窗模式)")
    print("'v' - Toggle landmarks (顯示關鍵點)")
    print("'m' - Change landmark mode (切換關鍵點模式: 1=全部 2=控制點 3=變形對比)")
    print("'g' - Toggle Golden Ratio Auto mode (黃金比例自動模式)")
    print("'i' - Toggle show ratios info (顯示比例資訊)")
    print("'b' - Toggle skin smooth (皮膚光滑開關)")
    print("'1-5' - Set smooth level (設定平滑等級)")
    print("'z/x' - Adjust detail preserve (細節保留)")
    print("'+/-' - Adjust cheek strength (手動模式)")
    print("'[/]' - Adjust chin strength (手動模式)")
    print("'</>' - Adjust golden ratio strength (黃金比例強度)")
    print("'s' - Switch warp method")
    print("'r/t' - Adjust grid resolution")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 360))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        annotated_image = np.copy(image.numpy_view())
        
        h, w = annotated_image.shape[:2]
        
        # STEP 4: Detect face landmarks from the input image.
        s = time.time()
        detection_result = detector.detect(image)
        
        # STEP 5: Process the detection result.
        face_landmarks_list = detection_result.face_landmarks
        
        out = annotated_image.copy()
        
        if len(face_landmarks_list) > 0:
            for idx in range(len(face_landmarks_list)):
                face_landmarks = face_landmarks_list[idx]
                
                # 1. 獲取臉部控制點
                src_points, indices, left_indices, right_indices, chin_indices = \
                    get_face_control_points(face_landmarks, w, h)
                
                # 1.5 計算黃金比例
                ratios_info = calculate_face_golden_ratios(face_landmarks, w, h)
                
                # 2. 應用網格變形（黃金比例模式或手動模式）
                if use_golden_ratio:
                    dst_points = apply_golden_ratio_deformation(
                        src_points, indices, left_indices, right_indices, chin_indices,
                        face_landmarks, w, h, ratios_info, golden_ratio_strength
                    )
                else:
                    dst_points = apply_mesh_deformation(
                        src_points, indices, left_indices, right_indices, chin_indices,
                        face_landmarks, w, h, cheek_strength, chin_strength
                    )
                
                # 3. 使用選定的變形方法
                if use_simple_warp:
                    out = simple_mesh_warp_image(annotated_image, src_points, dst_points, grid_resolution)
                else:
                    out = mesh_warp_image(annotated_image, src_points, dst_points)
                
                # 4. 應用皮膚光滑（在變形之後）
                if enable_skin_smooth and smooth_level > 0:
                    skin_mask = detect_skin_mask(out, face_landmarks, w, h)
                    out = smooth_skin(out, skin_mask, smooth_level, detail_preserve)
                
                # 5. 視覺化關鍵點
                if show_landmarks and landmark_mode > 0:
                    if landmark_mode == 1:
                        # 模式1：顯示所有468個關鍵點
                        for i, landmark in enumerate(face_landmarks):
                            x, y = to_pixel(landmark, w, h)
                            cv2.circle(out, (x, y), 1, (0, 255, 0), -1)
                            # 每隔N個點顯示編號（避免太擁擠）
                            if i % 20 == 0:
                                cv2.putText(out, str(i), (x+3, y-3), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                    
                    elif landmark_mode == 2:
                        # 模式2：只顯示控制點
                        for i, idx in enumerate(indices):
                            pt = src_points[i]
                            cv2.circle(out, tuple(pt.astype(int)), 3, (0, 255, 255), -1)
                            cv2.putText(out, str(idx), (int(pt[0])+5, int(pt[1])-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    elif landmark_mode == 3:
                        # 模式3：顯示變形前後對比
                        for pt in src_points:
                            cv2.circle(out, tuple(pt.astype(int)), 3, (0, 255, 0), -1)  # 綠色：原始點
                        for pt in dst_points:
                            cv2.circle(out, tuple(pt.astype(int)), 3, (0, 0, 255), -1)  # 紅色：目標點
                        # 畫線連接原始點和目標點
                        for i in range(len(src_points)):
                            pt1 = tuple(src_points[i].astype(int))
                            pt2 = tuple(dst_points[i].astype(int))
                            cv2.line(out, pt1, pt2, (255, 0, 255), 1)
        
        fps = 1.0 / (time.time() - s)
        
        # 顯示資訊
        y_offset = 30
        cv2.putText(out, f"FPS: {fps:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 35
        mode_text = f"Mode: {'Golden Ratio' if use_golden_ratio else 'Manual'}"
        cv2.putText(out, mode_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if use_golden_ratio else (255, 255, 0), 2)
        
        y_offset += 30
        if use_golden_ratio:
            cv2.putText(out, f"Golden Strength: {golden_ratio_strength:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(out, f"Cheek: {cheek_strength:.2f} Chin: {chin_strength:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        y_offset += 25
        smooth_text = f"Skin: {'ON' if enable_skin_smooth else 'OFF'}"
        if enable_skin_smooth:
            smooth_text += f" (Lvl:{smooth_level} Detail:{detail_preserve:.1f})"
        cv2.putText(out, smooth_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 255) if enable_skin_smooth else (128, 128, 128), 2)
        
        # 顯示關鍵點狀態
        if show_landmarks:
            y_offset += 25
            landmark_modes = ['OFF', 'All 468', 'Control', 'Deform']
            cv2.putText(out, f"Landmarks: {landmark_modes[landmark_mode]}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)
        
        # 顯示黃金比例資訊
        if show_ratios and len(face_landmarks_list) > 0:
            y_offset += 30
            cv2.putText(out, "--- Golden Ratios ---", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)
            y_offset += 25
            cv2.putText(out, f"W/H: {ratios_info['width_height_ratio']:.3f} (ideal: 0.618)", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(out, f"Thirds: {ratios_info['upper_third']:.0f}/{ratios_info['middle_third']:.0f}/{ratios_info['lower_third']:.0f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(out, f"Adj: Cheek={ratios_info['cheek_adjustment']:.2f} Chin={ratios_info['chin_adjustment']:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 顯示視窗
        if show_dual_view:
            # 雙視窗模式：原圖 + 處理後
            # 在原圖上添加標籤
            original_display = annotated_image.copy()
            cv2.putText(original_display, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(original_display, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 在處理後的圖上添加標籤
            processed_display = out.copy()
            cv2.putText(processed_display, "Processed", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # 顯示兩個視窗
            cv2.imshow("Original", original_display[:, :, ::-1])
            cv2.imshow("Processed", processed_display[:, :, ::-1])
        else:
            # 單視窗模式：只顯示處理後
            cv2.imshow("Face Beauty Filter", out[:, :, ::-1])
        
        # 鍵盤控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            cheek_strength = min(1.0, cheek_strength + 0.05)
            print(f"Cheek strength: {cheek_strength:.2f}")
        elif key == ord('-') or key == ord('_'):
            cheek_strength = max(0.0, cheek_strength - 0.05)
            print(f"Cheek strength: {cheek_strength:.2f}")
        elif key == ord('s'):
            use_simple_warp = not use_simple_warp
            print(f"Switched to: {'Fast RBF' if use_simple_warp else 'Triangle Mesh'}")
        elif key == ord('['):
            chin_strength = max(0.0, chin_strength - 0.05)
            print(f"Chin strength: {chin_strength:.2f}")
        elif key == ord(']'):
            chin_strength = min(1.0, chin_strength + 0.05)
            print(f"Chin strength: {chin_strength:.2f}")
        elif key == ord('r'):
            grid_resolution = max(20, grid_resolution - 5)
            print(f"Grid resolution: {grid_resolution} (faster)")
        elif key == ord('t'):
            grid_resolution = min(80, grid_resolution + 5)
            print(f"Grid resolution: {grid_resolution} (slower, better quality)")
        elif key == ord('g'):
            use_golden_ratio = not use_golden_ratio
            print(f"Golden Ratio Mode: {'ON' if use_golden_ratio else 'OFF'}")
        elif key == ord('i'):
            show_ratios = not show_ratios
            print(f"Show ratios info: {'ON' if show_ratios else 'OFF'}")
        elif key == ord(',') or key == ord('<'):
            golden_ratio_strength = max(0.0, golden_ratio_strength - 0.1)
            print(f"Golden ratio strength: {golden_ratio_strength:.2f}")
        elif key == ord('.') or key == ord('>'):
            golden_ratio_strength = min(1.0, golden_ratio_strength + 0.1)
            print(f"Golden ratio strength: {golden_ratio_strength:.2f}")
        elif key == ord('b'):
            enable_skin_smooth = not enable_skin_smooth
            print(f"Skin smooth: {'ON' if enable_skin_smooth else 'OFF'}")
        elif key >= ord('0') and key <= ord('5'):
            smooth_level = key - ord('0')
            print(f"Smooth level: {smooth_level}")
        elif key == ord('z'):
            detail_preserve = max(0.0, detail_preserve - 0.1)
            print(f"Detail preserve: {detail_preserve:.2f}")
        elif key == ord('x'):
            detail_preserve = min(1.0, detail_preserve + 0.1)
            print(f"Detail preserve: {detail_preserve:.2f}")
        elif key == ord('d'):
            show_dual_view = not show_dual_view
            # 關閉所有現有視窗
            cv2.destroyAllWindows()
            print(f"Dual view: {'ON' if show_dual_view else 'OFF'}")
        elif key == ord('v'):
            show_landmarks = not show_landmarks
            if not show_landmarks:
                landmark_mode = 0
            else:
                landmark_mode = 1
            print(f"Landmarks: {'ON' if show_landmarks else 'OFF'}")
        elif key == ord('m'):
            if show_landmarks:
                landmark_mode = (landmark_mode % 3) + 1
                landmark_modes = ['OFF', 'All 468 points', 'Control points', 'Deform comparison']
                print(f"Landmark mode: {landmark_modes[landmark_mode]}")
            else:
                print("Please enable landmarks first (press 'v')")
    
    cap.release()
    cv2.destroyAllWindows()
