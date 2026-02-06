import numpy as np
import cv2

# ---- your indices ----
RIGHT_CHEEK = [1, 9, 10, 11, 12, 13, 14, 15, 16]
LEFT_CHEEK  = [17, 25, 26, 27, 28, 29, 30, 31, 32]
CHIN        = [2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18]
NOSE        = [73, 74, 76, 77, 78, 79, 80, 85, 84, 83, 82, 86]
RIGHT_EYE   = [35, 36, 33, 37, 39, 42, 40, 41, 38]
LEFT_EYE    = [93, 96, 94, 95, 89, 90, 87, 91, 88]
MOUTH       = [71, 63, 64, 52, 55, 56, 53, 59, 58, 61, 68, 67, 62, 66, 65, 54, 60, 57, 69, 70]
RIGHT_EYEBROWS = [43, 48, 49, 51, 50, 46, 47, 45, 44]
LEFT_EYEBROWS  = [101, 105, 104, 103, 102, 97, 98, 99, 100]
BETWEEN_EYES   = [75, 72, 81]


# ---------- helpers ----------
def _bilinear_sample(img, x, y):
    """img: HxWxC, x/y: HxW float maps (source coords). returns warped image"""
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # expand weights to channels
    if img.ndim == 3:
        wa = wa[..., None]
        wb = wb[..., None]
        wc = wc[..., None]
        wd = wd[..., None]

    out = Ia * wa + Ib * wb + Ic * wc + Id * wd
    return out


def _face_mask_from_landmarks(img_shape, lm, extra_idx=None):
    """用 landmarks hull 當臉部 mask，避免背景被扯"""
    h, w = img_shape[:2]
    idx = []
    idx += CHIN
    idx += LEFT_CHEEK
    idx += RIGHT_CHEEK
    if extra_idx is not None:
        idx += extra_idx

    pts = lm[np.array(idx, dtype=np.int32)].astype(np.int32)
    hull = cv2.convexHull(pts)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillConvexPoly(mask, hull, 1.0)
    # 稍微羽化，邊緣更自然
    mask = cv2.GaussianBlur(mask, (0, 0), 3)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _radial_eye_enlarge_map(h, w, center, radius, strength):
    """
    inverse mapping: for each output pixel p, compute source p'
    enlarge effect via pulling samples toward center inside radius
    """
    cx, cy = center
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    d2 = dx*dx + dy*dy
    r2 = radius * radius

    inside = d2 < r2
    d = np.sqrt(np.maximum(d2, 1e-6))
    t = d / (radius + 1e-6)

    # smooth weight: (1 - t^2)^2
    wgt = (1 - t*t)
    wgt = wgt * wgt
    wgt *= inside.astype(np.float32)

    # scale < 1 => sample closer to center => appears enlarged
    scale = 1.0 - strength * wgt
    map_x = cx + dx * scale
    map_y = cy + dy * scale
    return map_x, map_y, wgt


def _point_to_point_map(h, w, q, q_prime, radius):
    """
    inverse mapping for slim face:
    want q moves to q', so for output p, source = p - w*(q' - q)
    """
    qx, qy = q
    dx_move = (q_prime[0] - qx)
    dy_move = (q_prime[1] - qy)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - qx
    dy = yy - qy
    d2 = dx*dx + dy*dy
    r2 = radius * radius

    inside = d2 < r2
    d = np.sqrt(np.maximum(d2, 1e-6))
    t = d / (radius + 1e-6)

    wgt = (1 - t*t)
    wgt = wgt * wgt
    wgt *= inside.astype(np.float32)

    map_x = xx - wgt * dx_move
    map_y = yy - wgt * dy_move
    return map_x, map_y, wgt


def _pick_cheek_point(lm, cheek_idx, face_center):
    """
    從 cheek control points 選一個代表點：
    用「離臉中心最遠」通常比單純 min/max x 穩
    """
    pts = lm[np.array(cheek_idx, dtype=np.int32)]
    fc = np.array(face_center, dtype=np.float32)[None, :]
    dist2 = np.sum((pts - fc) ** 2, axis=1)
    return pts[np.argmax(dist2)]


def _eye_center_and_width(lm, eye_idx):
    pts = lm[np.array(eye_idx, dtype=np.int32)].astype(np.float32)
    c = pts.mean(axis=0)
    # 眼寬用 x 範圍（也可以用兩眼角點距離）
    eye_w = (pts[:, 0].max() - pts[:, 0].min())
    eye_h = (pts[:, 1].max() - pts[:, 1].min())
    scale = max(eye_w, eye_h, 1.0)
    return c, scale


# ---------- main beautify ----------
def beautify_slim_bigeye(
    img_bgr: np.ndarray,
    landmarks106: np.ndarray,
    slim_strength: float = 0.12,     # 0~0.2 建議
    eye_strength: float = 0.22,      # 0~0.35 建議
    cheek_radius_ratio: float = 0.45,# face_width 的比例
    eye_radius_ratio: float = 2.0,   # eye_size 的倍率
):
    """
    img_bgr: uint8 HxWx3
    landmarks106: (106,2)
    returns: beautified img
    """
    img = img_bgr.astype(np.float32)
    lm = landmarks106.astype(np.float32)
    h, w = img.shape[:2]

    # face center：用 BETWEEN_EYES + NOSE 的平均更穩
    center_pts = np.concatenate([lm[np.array(BETWEEN_EYES)], lm[np.array(NOSE)]], axis=0)
    face_center = center_pts.mean(axis=0)

    # face width: 用左右臉頰代表點的距離
    left_q  = _pick_cheek_point(lm, LEFT_CHEEK,  face_center)
    right_q = _pick_cheek_point(lm, RIGHT_CHEEK, face_center)
    face_w = np.linalg.norm(left_q - right_q) + 1e-6

    # 臉部mask（避免背景/頭髮被拉）
    face_mask = _face_mask_from_landmarks(img.shape, lm)

    # 先做大眼（通常先眼睛效果比較自然）
    out = img.copy()
    for eye_idx in [LEFT_EYE, RIGHT_EYE]:
        c, eye_scale = _eye_center_and_width(lm, eye_idx)
        R = eye_radius_ratio * eye_scale
        map_x, map_y, wgt = _radial_eye_enlarge_map(h, w, c, R, eye_strength)

        warped = _bilinear_sample(out, map_x, map_y)
        # 限制在臉mask + 眼睛局部權重（wgt）
        local_mask = (face_mask * wgt).astype(np.float32)
        local_mask = local_mask[..., None]
        out = out * (1 - local_mask) + warped * local_mask

    # 再做瘦臉（左右臉頰往內）
    cheek_R = cheek_radius_ratio * face_w

    # 左臉頰往 face_center 拉
    q = left_q
    q_prime = q + slim_strength * (face_center - q)
    map_x, map_y, wgt = _point_to_point_map(h, w, q, q_prime, cheek_R)
    warped = _bilinear_sample(out, map_x, map_y)
    local_mask = (face_mask * wgt).astype(np.float32)[..., None]
    out = out * (1 - local_mask) + warped * local_mask

    # 右臉頰往 face_center 拉
    q = right_q
    q_prime = q + slim_strength * (face_center - q)
    map_x, map_y, wgt = _point_to_point_map(h, w, q, q_prime, cheek_R)
    warped = _bilinear_sample(out, map_x, map_y)
    local_mask = (face_mask * wgt).astype(np.float32)[..., None]
    out = out * (1 - local_mask) + warped * local_mask

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

if __name__ == "__main__":
    
    from insightface.app import FaceAnalysis
    
    # Initialize InsightFace
    print("Initializing InsightFace...")
    model = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=["detection", "landmark_2d_106"],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    
    # ---- test code ----
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Detect faces
        faces = model.get(frame)
        for face in faces:
            landmarks106 = face.get("landmark_2d_106")
            out = beautify_slim_bigeye(frame, landmarks106)
        cv2.imshow("input", frame)
        cv2.imshow("output", out)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()