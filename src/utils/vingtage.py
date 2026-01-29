import numpy as np
import cv2

def build_fade_curve(contrast=0.12, lift=12, gamma=1.05):
    # ...existing code...
    x = np.arange(256).astype(np.float32)
    y = 255.0 * ((x / 255.0) ** (1.0 / gamma))
    t = (y - 128.0) / 128.0
    y = 128.0 + 128.0 * np.tanh((1.0 + contrast*6.0) * t)
    y = y * (255 - lift) / 255.0 + lift
    return np.clip(y, 0, 255).astype(np.uint8)

def apply_curve_bgr(bgr, curve):
    return cv2.LUT(bgr, curve)

# 快取 vignette mask
_vignette_cache = {}

def vignette(bgr, strength=0.35):
    """
    使用快取避免重複計算 mask
    """
    h, w = bgr.shape[:2]
    key = (h, w, strength)
    
    if key not in _vignette_cache:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w * 0.5, h * 0.5
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        mask = 1.0 - strength * (r ** 1.6)
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
        # 轉成 3 通道並用 uint8 LUT 方式優化
        _vignette_cache[key] = np.stack([mask]*3, axis=-1)
    
    mask = _vignette_cache[key]
    # 使用 cv2.multiply 比 numpy 更快
    return cv2.multiply(bgr.astype(np.float32), mask, dtype=cv2.CV_8U)