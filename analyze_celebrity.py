"""
分析明星照片的臉部比例
使用方法：python analyze_celebrity.py <照片路徑>
"""
import cv2
import sys
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json

def to_pixel(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def extract_face_proportions(face_landmarks, w, h):
    """提取臉部關鍵比例"""
    # MediaPipe 468 個關鍵點
    LEFT_FACE = 234    # 左臉頰
    RIGHT_FACE = 454   # 右臉頰
    TOP_HEAD = 10      # 額頭頂部
    CHIN = 152         # 下巴底部
    LEFT_JAW = 172     # 左下顎
    RIGHT_JAW = 397    # 右下顎
    
    # 轉換為像素座標
    left_face = to_pixel(face_landmarks[LEFT_FACE], w, h)
    right_face = to_pixel(face_landmarks[RIGHT_FACE], w, h)
    top_head = to_pixel(face_landmarks[TOP_HEAD], w, h)
    chin = to_pixel(face_landmarks[CHIN], w, h)
    left_jaw = to_pixel(face_landmarks[LEFT_JAW], w, h)
    right_jaw = to_pixel(face_landmarks[RIGHT_JAW], w, h)
    
    # 計算距離
    face_width = np.linalg.norm(np.array(left_face) - np.array(right_face))
    face_height = np.linalg.norm(np.array(top_head) - np.array(chin))
    jaw_width = np.linalg.norm(np.array(left_jaw) - np.array(right_jaw))
    
    # 計算比例
    face_ratio = face_width / (face_height + 1e-6)
    jaw_ratio = jaw_width / (face_width + 1e-6)
    
    proportions = {
        'face_width': float(face_width),
        'face_height': float(face_height),
        'jaw_width': float(jaw_width),
        'cheekbone_width': float(face_width),
        'face_ratio': float(face_ratio),
        'jaw_ratio': float(jaw_ratio),
    }
    
    return proportions

def analyze_celebrity_face(image_path):
    """分析明星照片並顯示結果"""
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
    
    # 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"錯誤：無法讀取圖片 {image_path}")
        return None
    
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # 檢測臉部
    detection_result = detector.detect(mp_image)
    
    if not detection_result.face_landmarks:
        print("錯誤：未能在照片中檢測到臉部")
        return None
    
    face_landmarks = detection_result.face_landmarks[0]
    proportions = extract_face_proportions(face_landmarks, w, h)
    
    # 在圖片上標註關鍵點
    annotated_img = img.copy()
    
    # 標註關鍵點
    LEFT_FACE = 234
    RIGHT_FACE = 454
    TOP_HEAD = 10
    CHIN = 152
    LEFT_JAW = 172
    RIGHT_JAW = 397
    NOSE_TIP = 1
    
    points = {
        'Left Face': (LEFT_FACE, (255, 0, 0)),
        'Right Face': (RIGHT_FACE, (0, 0, 255)),
        'Top Head': (TOP_HEAD, (0, 255, 0)),
        'Chin': (CHIN, (255, 255, 0)),
        'Left Jaw': (LEFT_JAW, (255, 0, 255)),
        'Right Jaw': (RIGHT_JAW, (0, 255, 255)),
        'Nose': (NOSE_TIP, (0, 255, 0))
    }
    
    for name, (idx, color) in points.items():
        x, y = to_pixel(face_landmarks[idx], w, h)
        cv2.circle(annotated_img, (x, y), 5, color, -1)
        cv2.putText(annotated_img, name, (x + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # 顯示結果
    print("\n" + "="*50)
    print("臉部比例分析結果")
    print("="*50)
    print(f"臉寬 (像素): {proportions['face_width']:.1f}")
    print(f"臉長 (像素): {proportions['face_height']:.1f}")
    print(f"下巴寬度 (像素): {proportions['jaw_width']:.1f}")
    print(f"\n臉寬/臉長比例: {proportions['face_ratio']:.3f}")
    print(f"下巴/臉寬比例: {proportions['jaw_ratio']:.3f}")
    print("="*50 + "\n")
    
    # 保存標註圖片
    output_path = image_path.replace('.', '_analyzed.')
    cv2.imwrite(output_path, annotated_img)
    print(f"已保存標註圖片到: {output_path}")
    
    # 保存比例數據
    json_path = image_path.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(proportions, f, indent=2, ensure_ascii=False)
    print(f"已保存比例數據到: {json_path}")
    
    # 顯示圖片
    cv2.imshow("Celebrity Face Analysis (Press any key to close)", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return proportions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python analyze_celebrity.py <照片路徑>")
        print("範例: python analyze_celebrity.py .assets/images/celebrity.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    analyze_celebrity_face(image_path)
