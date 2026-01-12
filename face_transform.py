"""
基於明星臉部比例的臉部變換系統
使用 Delaunay 三角變換技術
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os

def to_pixel(lm, w, h):
    """將 MediaPipe 標準化座標轉換為像素座標"""
    return (int(lm.x * w), int(lm.y * h))

def extract_key_landmarks(face_landmarks, w, h):
    """提取關鍵臉部特徵點（用於三角變換）"""
    # 選擇重要的特徵點（減少計算量但保持精度）
    key_indices = [
        # 臉部輪廓
        10, 338, 297, 332, 284, 251, 389, 356, 454,  # 右側輪廓
        323, 361, 288, 397, 365, 379, 378, 400, 377, 152,  # 下巴
        150, 176, 148, 172, 136, 58, 132, 93, 234,  # 左側輪廓
        127, 162, 21, 54, 103, 67, 109,  # 左額頭-臉頰
        
        # 鼻子
        1, 4, 5, 6, 168, 195, 197,
        
        # 眼睛
        33, 133, 160, 159, 158, 157, 173, 246,  # 左眼
        362, 263, 387, 386, 385, 384, 398, 466,  # 右眼
        
        # 嘴巴
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        
        # 眉毛
        70, 63, 105, 66, 107,  # 左眉
        336, 296, 334, 293, 300,  # 右眉
    ]
    
    points = []
    for idx in key_indices:
        if idx < len(face_landmarks):
            x, y = to_pixel(face_landmarks[idx], w, h)
            points.append([x, y])
    
    return np.array(points, dtype=np.float32), key_indices

def calculate_face_metrics(face_landmarks, w, h):
    """計算當前臉部的關鍵測量數據"""
    LEFT_FACE = 234
    RIGHT_FACE = 454
    TOP_HEAD = 10
    CHIN = 152
    LEFT_JAW = 172
    RIGHT_JAW = 397
    
    left_face = to_pixel(face_landmarks[LEFT_FACE], w, h)
    right_face = to_pixel(face_landmarks[RIGHT_FACE], w, h)
    top_head = to_pixel(face_landmarks[TOP_HEAD], w, h)
    chin = to_pixel(face_landmarks[CHIN], w, h)
    left_jaw = to_pixel(face_landmarks[LEFT_JAW], w, h)
    right_jaw = to_pixel(face_landmarks[RIGHT_JAW], w, h)
    
    face_width = np.linalg.norm(np.array(left_face) - np.array(right_face))
    face_height = np.linalg.norm(np.array(top_head) - np.array(chin))
    jaw_width = np.linalg.norm(np.array(left_jaw) - np.array(right_jaw))
    
    return {
        'face_width': float(face_width),
        'face_height': float(face_height),
        'jaw_width': float(jaw_width),
        'face_ratio': float(face_width / (face_height + 1e-6)),
        'jaw_ratio': float(jaw_width / (face_width + 1e-6)),
    }

def adjust_landmarks_to_target_proportions(face_landmarks, w, h, target_proportions, intensity=0.5):
    """
    根據目標比例調整臉部特徵點
    intensity: 0~1，調整強度（0=不變，1=完全變成目標比例）
    """
    current_metrics = calculate_face_metrics(face_landmarks, w, h)
    
    # 計算需要調整的比例
    width_scale = 1.0
    jaw_scale = 1.0
    
    # 臉寬調整（基於 face_ratio）
    target_ratio = target_proportions.get('face_ratio', current_metrics['face_ratio'])
    current_ratio = current_metrics['face_ratio']
    if target_ratio < current_ratio:
        # 需要變窄
        width_scale = 1.0 - (current_ratio - target_ratio) * intensity
    
    # 下巴寬度調整
    target_jaw_ratio = target_proportions.get('jaw_ratio', current_metrics['jaw_ratio'])
    current_jaw_ratio = current_metrics['jaw_ratio']
    if target_jaw_ratio < current_jaw_ratio:
        # 需要變尖
        jaw_scale = 1.0 - (current_jaw_ratio - target_jaw_ratio) * intensity * 1.5
    
    # 獲取臉部中心線（鼻子）
    NOSE_CENTER = 1
    nose_x = face_landmarks[NOSE_CENTER].x * w
    
    # 調整所有特徵點
    adjusted_landmarks = []
    for i, lm in enumerate(face_landmarks):
        x, y = lm.x * w, lm.y * h
        
        # 水平方向：離中心線越遠，縮放越明顯
        offset_x = x - nose_x
        
        # 臉頰區域（左右兩側）使用 width_scale
        if abs(offset_x) > w * 0.1:  # 不是中心區域
            # 計算縮放因子（下半臉更明顯）
            y_factor = max(0, (y - h * 0.3) / (h * 0.7))  # 0~1，越往下越大
            scale = width_scale + (jaw_scale - width_scale) * y_factor
            new_offset_x = offset_x * scale
            x = nose_x + new_offset_x
        
        adjusted_landmarks.append((x, y))
    
    return adjusted_landmarks

def apply_delaunay_warp(img, src_points, dst_points):
    """
    使用 Delaunay 三角變換進行臉部變形
    """
    h, w = img.shape[:2]
    
    # 添加圖片四個角作為邊界點
    corners = np.array([
        [0, 0], [w-1, 0], [w-1, h-1], [0, h-1],
        [w//2, 0], [w-1, h//2], [w//2, h-1], [0, h//2]
    ], dtype=np.float32)
    
    src_points_with_corners = np.vstack([src_points, corners])
    dst_points_with_corners = np.vstack([dst_points, corners])
    
    # 計算 Delaunay 三角剖分
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    
    for p in src_points_with_corners:
        subdiv.insert((float(p[0]), float(p[1])))
    
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    
    # 創建輸出圖像
    output = np.zeros_like(img)
    
    # 對每個三角形進行仿射變換
    for t in triangles:
        # 三角形三個頂點
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        # 找到這些點在源點集中的索引
        tri_src = []
        tri_dst = []
        
        for pt in [pt1, pt2, pt3]:
            # 找最近的點
            distances = np.linalg.norm(src_points_with_corners - np.array(pt), axis=1)
            idx = np.argmin(distances)
            if distances[idx] < 5:  # 距離閾值
                tri_src.append(src_points_with_corners[idx])
                tri_dst.append(dst_points_with_corners[idx])
        
        if len(tri_src) == 3:
            tri_src = np.array(tri_src, dtype=np.float32)
            tri_dst = np.array(tri_dst, dtype=np.float32)
            
            # 執行三角形變形
            warp_triangle(img, output, tri_src, tri_dst)
    
    return output

def warp_triangle(img_src, img_dst, tri_src, tri_dst):
    """
    對單個三角形執行仿射變換
    """
    # 獲取三角形的邊界矩形
    rect_src = cv2.boundingRect(tri_src)
    rect_dst = cv2.boundingRect(tri_dst)
    
    # 將三角形座標轉換為相對於邊界矩形的座標
    tri_src_rect = []
    tri_dst_rect = []
    
    for i in range(3):
        tri_src_rect.append(((tri_src[i][0] - rect_src[0]), (tri_src[i][1] - rect_src[1])))
        tri_dst_rect.append(((tri_dst[i][0] - rect_dst[0]), (tri_dst[i][1] - rect_dst[1])))
    
    # 獲取變換矩陣
    try:
        mat = cv2.getAffineTransform(np.float32(tri_src_rect), np.float32(tri_dst_rect))
    except:
        return
    
    # 提取源三角形區域
    src_patch = img_src[rect_src[1]:rect_src[1]+rect_src[3], 
                        rect_src[0]:rect_src[0]+rect_src[2]]
    
    if src_patch.size == 0:
        return
    
    # 執行仿射變換
    size = (rect_dst[2], rect_dst[3])
    dst_patch = cv2.warpAffine(src_patch, mat, size, 
                               flags=cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_REFLECT_101)
    
    # 創建遮罩
    mask = np.zeros((rect_dst[3], rect_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tri_dst_rect), (1.0, 1.0, 1.0))
    
    # 將變形後的三角形混合到目標圖像
    try:
        roi = img_dst[rect_dst[1]:rect_dst[1]+rect_dst[3], 
                      rect_dst[0]:rect_dst[0]+rect_dst[2]]
        roi[:] = roi * (1 - mask) + dst_patch * mask
    except:
        pass

def transform_face_with_celebrity_proportions(
    image_path, 
    celebrity_json_path, 
    output_path="transformed_face.jpg",
    intensity=0.5,
    show_landmarks=False
):
    """
    主函數：將照片中的臉部按明星比例變換
    
    參數：
    - image_path: 要處理的照片路徑
    - celebrity_json_path: 明星臉部比例 JSON 檔案路徑
    - output_path: 輸出檔案路徑
    - intensity: 變換強度 (0~1)
    - show_landmarks: 是否顯示特徵點
    """
    # 讀取明星比例
    if not os.path.exists(celebrity_json_path):
        print(f"錯誤：找不到明星比例檔案 {celebrity_json_path}")
        return None
    
    with open(celebrity_json_path, 'r', encoding='utf-8') as f:
        celebrity_proportions = json.load(f)
    
    print(f"\n已載入明星比例：")
    print(f"  - 臉寬/臉長比例: {celebrity_proportions['face_ratio']:.3f}")
    print(f"  - 下巴/臉寬比例: {celebrity_proportions['jaw_ratio']:.3f}")
    
    # 創建 FaceLandmarker
    model_path = ".assets/models/face_landmarker_v2_with_blendshapes.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    # 讀取照片
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取照片 {image_path}")
        return None
    
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # 檢測臉部
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        print("錯誤：照片中未檢測到臉部")
        return None
    
    face_landmarks = detection_result.face_landmarks[0]
    
    # 計算當前臉部比例
    current_metrics = calculate_face_metrics(face_landmarks, w, h)
    print(f"\n當前臉部比例：")
    print(f"  - 臉寬/臉長比例: {current_metrics['face_ratio']:.3f}")
    print(f"  - 下巴/臉寬比例: {current_metrics['jaw_ratio']:.3f}")
    
    # 調整特徵點到目標比例
    adjusted_landmarks = adjust_landmarks_to_target_proportions(
        face_landmarks, w, h, celebrity_proportions, intensity
    )
    
    # 提取關鍵特徵點用於三角變換
    src_points, key_indices = extract_key_landmarks(face_landmarks, w, h)
    
    # 構建目標點（調整後的位置）
    dst_points = []
    for idx in key_indices:
        if idx < len(adjusted_landmarks):
            dst_points.append(adjusted_landmarks[idx])
    dst_points = np.array(dst_points, dtype=np.float32)
    
    print(f"\n使用 {len(src_points)} 個特徵點進行 Delaunay 三角變換...")
    
    # 執行 Delaunay 三角變換
    result = apply_delaunay_warp(img, src_points, dst_points)
    
    # 顯示特徵點（可選）
    if show_landmarks:
        for pt in dst_points:
            cv2.circle(result, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
    
    # 保存結果
    cv2.imwrite(output_path, result)
    print(f"\n✅ 已保存變換結果到: {output_path}")
    
    # 顯示結果
    comparison = np.hstack([img, result])
    comparison = cv2.resize(comparison, (1600, 800))
    cv2.imshow("Before (Left) vs After (Right) - Press any key to close", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法:")
        print("  python face_transform.py <照片路徑> <明星比例JSON> [強度] [輸出路徑]")
        print("\n範例:")
        print("  python face_transform.py my_photo.jpg celebrity.json 0.5 output.jpg")
        print("\n參數說明:")
        print("  - 強度: 0~1 之間，預設 0.5")
        sys.exit(1)
    
    image_path = sys.argv[1]
    celebrity_json = sys.argv[2]
    intensity = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    output_path = sys.argv[4] if len(sys.argv) > 4 else "transformed_face.jpg"
    
    transform_face_with_celebrity_proportions(
        image_path, 
        celebrity_json, 
        output_path, 
        intensity,
        show_landmarks=False
    )
