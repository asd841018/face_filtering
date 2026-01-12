# STEP 1: Import the necessary modules.
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

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
    
    

def local_shrink(img, center, radius, strength, direction_to):
    """
    以 center 為中心、半徑 radius 的區域，往 direction_to 方向內縮。
    strength: 0~1, 越大越瘦
    """
    h, w = img.shape[:2]
    cx, cy = center
    tx, ty = direction_to

    # ROI 範圍
    x0 = max(cx - radius, 0)
    y0 = max(cy - radius, 0)
    x1 = min(cx + radius, w - 1)
    y1 = min(cy + radius, h - 1)

    roi = img[y0:y1, x0:x1].copy()
    rh, rw = roi.shape[:2]

    # 建立座標網格
    grid_x, grid_y = np.meshgrid(np.arange(rw), np.arange(rh))
    map_x = grid_x.astype(np.float32)
    map_y = grid_y.astype(np.float32)

    # 將 ROI 座標轉成全圖座標，再算向量
    abs_x = grid_x + x0
    abs_y = grid_y + y0

    dx = abs_x - cx
    dy = abs_y - cy
    dist = np.sqrt(dx * dx + dy * dy) + 1e-6

    # ROI 內、半徑範圍才做形變
    mask = dist < radius

    # 往目標方向（通常是往鼻子）推
    vtx = tx - abs_x
    vty = ty - abs_y
    vtx = vtx.astype(np.float32)
    vty = vty.astype(np.float32)
    vnorm = np.sqrt(vtx * vtx + vty * vty) + 1e-6
    vtx /= vnorm
    vty /= vnorm

    # 距離越靠近中心影響越大（你也可以換成更柔的曲線）
    falloff = (1.0 - (dist / radius)) ** 2

    # 位移量：strength * falloff * radius * 常數
    # 這個常數 0.35 可自行調整（越大越誇張）
    shift = strength * falloff * radius * 0.35

    # remap 是「從哪裡取像素」：想把當前點往目標方向推，
    # 就要從反方向取樣（減少拉扯破洞）
    map_x[mask] = (grid_x[mask] - vtx[mask] * shift[mask]).astype(np.float32)
    map_y[mask] = (grid_y[mask] - vty[mask] * shift[mask]).astype(np.float32)

    warped = cv2.remap(roi, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    out = img.copy()
    out[y0:y1, x0:x1] = warped
    return out
    
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

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(".assets/images/male3.jpg")
    annotated_image = np.copy(image.numpy_view())
    h, w = annotated_image.shape[:2]

    # Params for face slimming
    strength = 0.35  # 0~1
    radius_ratio = 0.15  # 臉頰變形半徑相對於影像寬度比例
    
    # 這幾個 landmark 點大致落在臉頰外側（MediaPipe 468 landmarks）
    # 左臉頰外側、右臉頰外側（靠近顴骨/臉側）
    CHEEK_LEFT = 234
    CHEEK_RIGHT = 454

    # 鼻尖附近，當作臉的中心參考（也可換成 1 或 4）
    NOSE_TIP = 1
    landmark_idx = [CHEEK_LEFT, CHEEK_RIGHT, NOSE_TIP]

    # STEP 4: Detect face landmarks from the input image.
    s = time.time()
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result.
    face_landmarks_list = detection_result.face_landmarks
    
    
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
        nose = to_pixel(face_landmarks[NOSE_TIP], annotated_image.shape[1], annotated_image.shape[0])
        cheek_left = to_pixel(face_landmarks[CHEEK_LEFT], annotated_image.shape[1], annotated_image.shape[0])
        cheek_right = to_pixel(face_landmarks[CHEEK_RIGHT], annotated_image.shape[1], annotated_image.shape[0])
        
        radius = int(w * radius_ratio)
        
         # 左右臉頰往鼻子方向縮
        out = local_shrink(annotated_image, center=cheek_left, radius=radius, strength=strength, direction_to=nose)
        out = local_shrink(out, center=cheek_right, radius=radius, strength=strength, direction_to=nose)

        # 顯示輔助點（可註解）
        cv2.circle(out, nose, 3, (0, 255, 0), -1)
        cv2.circle(out, cheek_left, 3, (255, 0, 0), -1)
        cv2.circle(out, cheek_right, 3, (0, 0, 255), -1)
        print(f"FPS: {1.0 / (time.time() - s):.2f}")
        
        
        
        # for i in landmark_idx:
        #     x = int(face_landmarks[i].x * annotated_image.shape[1])
        #     y = int(face_landmarks[i].y * annotated_image.shape[0])
        #     cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
        # for face_landmark in face_landmarks:
        #     x = int(face_landmark.x * annotated_image.shape[1])
        #     y = int(face_landmark.y * annotated_image.shape[0])
        #     cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)
                
    cv2.imwrite("annotated_image.jpg", out[:, :, ::-1])