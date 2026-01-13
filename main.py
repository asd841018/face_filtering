# STEP 1: Import the necessary modules.
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.interpolate import Rbf

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    print(face_blendshapes)
    # print("Face Blendshapes:")
    # for i in face_blendshapes_ranks:
    #     print(f"{face_blendshapes_names[i]}: {face_blendshapes_scores[i]:.4f}")
    
    

def get_cheek_contour_points(face_landmarks, w, h):
    """
    1. 偵測臉部關鍵點（已由 MediaPipe 完成）
    2. 選定臉頰輪廓點
    返回左右臉頰的關鍵點索引和像素座標
    """
    # MediaPipe 468 landmarks 中的臉頰輪廓點
    # 左臉頰輪廓（從顴骨到下巴）
    LEFT_CHEEK_CONTOUR = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
    # 右臉頰輪廓（從顴骨到下巴）
    RIGHT_CHEEK_CONTOUR = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]
    
    # 鼻子和臉部中心點作為參考
    CENTER_POINTS = [1, 4, 5, 6, 168, 197, 195]
    
    all_indices = LEFT_CHEEK_CONTOUR + RIGHT_CHEEK_CONTOUR + CENTER_POINTS
    
    src_points = []
    for idx in all_indices:
        x, y = to_pixel(face_landmarks[idx], w, h)
        src_points.append([x, y])
    
    return np.array(src_points), all_indices, LEFT_CHEEK_CONTOUR, RIGHT_CHEEK_CONTOUR

def offset_points_inward(src_points, indices, left_indices, right_indices, face_landmarks, w, h, offset_strength=0.3):
    """
    3. 對臉頰輪廓點往內偏移
    offset_strength: 偏移強度（0~1）
    """
    dst_points = src_points.copy()
    
    # 取得鼻尖作為中心參考點
    nose_x, nose_y = to_pixel(face_landmarks[1], w, h)
    
    # 對左臉頰點進行偏移（往鼻子方向）
    for i, idx in enumerate(left_indices):
        orig_idx = indices.index(idx)
        x, y = src_points[orig_idx]
        # 計算往鼻子方向的向量
        dx = nose_x - x
        dy = nose_y - y
        # 應用偏移
        dst_points[orig_idx][0] = x + dx * offset_strength
        dst_points[orig_idx][1] = y + dy * offset_strength
    
    # 對右臉頰點進行偏移（往鼻子方向）
    for i, idx in enumerate(right_indices):
        orig_idx = indices.index(idx)
        x, y = src_points[orig_idx]
        # 計算往鼻子方向的向量
        dx = nose_x - x
        dy = nose_y - y
        # 應用偏移
        dst_points[orig_idx][0] = x + dx * offset_strength
        dst_points[orig_idx][1] = y + dy * offset_strength
    
    return dst_points

def tps_warp_image(img, src_points, dst_points, grid_resolution=50):
    """
    4. 使用 TPS (Thin Plate Spline) warp 整張圖（優化速度版）
    添加邊界控制點以固定圖片邊緣，避免背景扭曲
    grid_resolution: 網格解析度（越小越快，但精度降低）
    """
    h, w = img.shape[:2]
    
    # 在圖片四周添加邊界控制點（保持不變）
    border_points = []
    margin = max(50, min(w, h) // 10)  # 增大間距以減少控制點數量
    
    # 只在四個角和邊界的幾個點
    # 四個角
    border_points.extend([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    
    # 上下邊界
    for x in range(margin, w-margin, margin):
        border_points.extend([[x, 0], [x, h-1]])
    
    # 左右邊界
    for y in range(margin, h-margin, margin):
        border_points.extend([[0, y], [w-1, y]])
    
    border_points = np.array(border_points)
    
    # 合併臉部控制點和邊界點
    all_src_points = np.vstack([src_points, border_points])
    all_dst_points = np.vstack([dst_points, border_points])
    
    # 建立 TPS 映射（使用降採樣網格加速計算）
    rbf_x = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 0], 
                function='thin_plate', smooth=0)
    rbf_y = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 1], 
                function='thin_plate', smooth=0)
    
    # 使用較低解析度的網格計算，然後插值放大（大幅提升速度）
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
    
def to_pixel(lm, w, h):
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
    # Params for face slimming
    offset_strength = 0.10  # 往內偏移強度 (0~1)，越大臉越瘦
    
    # image = mp.Image.create_from_file(".assets/images/male3.jpg")
    # STEP 3: Load the input image.
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
        annotated_image = np.copy(image.numpy_view())
        h, w = annotated_image.shape[:2]

        # STEP 4: Detect face landmarks from the input image.
        s = time.time()
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result.
        face_landmarks_list = detection_result.face_landmarks
        
        out = annotated_image.copy()
        
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            
            # 1. 偵測臉部關鍵點（已由 MediaPipe 完成）
            # 2. 選定臉頰輪廓點
            src_points, indices, left_indices, right_indices = get_cheek_contour_points(
                face_landmarks, w, h
            )
            
            # 3. 對點往內偏移
            dst_points = offset_points_inward(
                src_points, indices, left_indices, right_indices,
                face_landmarks, w, h, offset_strength
            )
            
            # 4. TPS warp 整張圖
            out = tps_warp_image(annotated_image, src_points, dst_points)
            
            # 視覺化源點和目標點（可選）
            for pt in src_points:
                cv2.circle(out, tuple(pt.astype(int)), 2, (0, 255, 0), -1)  # 綠色：原始點
            for pt in dst_points:
                cv2.circle(out, tuple(pt.astype(int)), 2, (0, 0, 255), -1)  # 紅色：目標點
        
        print(f"FPS: {1.0 / (time.time() - s):.2f}")
        cv2.imshow("Face Slimming", out[:, :, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

                
    # cv2.imwrite("annotated_image.jpg", out[:, :, ::-1])