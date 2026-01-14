import numpy as np
import cv2

def create_identity_lut(N=33):
    lut = np.zeros((N, N, N, 3), dtype=np.float32)
    for r in range(N):
        for g in range(N):
            for b in range(N):
                lut[r, g, b] = [
                    r / (N - 1),
                    g / (N - 1),
                    b / (N - 1)
                ]
    return lut

def apply_3d_lut(image, lut):
    """
    image: uint8 BGR image
    lut:   float32 LUT with shape (N, N, N, 3), range [0,1]
    """
    N = lut.shape[0]
    img = image.astype(np.float32) / 255.0

    # OpenCV 是 BGR
    b = img[..., 0] * (N - 1)
    g = img[..., 1] * (N - 1)
    r = img[..., 2] * (N - 1)

    r0 = np.floor(r).astype(int)
    g0 = np.floor(g).astype(int)
    b0 = np.floor(b).astype(int)

    r1 = np.clip(r0 + 1, 0, N - 1)
    g1 = np.clip(g0 + 1, 0, N - 1)
    b1 = np.clip(b0 + 1, 0, N - 1)

    dr = r - r0
    dg = g - g0
    db = b - b0

    # 8 個 cube 頂點
    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    c00 = c000 * (1 - db)[..., None] + c001 * db[..., None]
    c01 = c010 * (1 - db)[..., None] + c011 * db[..., None]
    c10 = c100 * (1 - db)[..., None] + c101 * db[..., None]
    c11 = c110 * (1 - db)[..., None] + c111 * db[..., None]

    c0 = c00 * (1 - dg)[..., None] + c01 * dg[..., None]
    c1 = c10 * (1 - dg)[..., None] + c11 * dg[..., None]

    out = c0 * (1 - dr)[..., None] + c1 * dr[..., None]

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


if __name__ == "__main__":
    
    cap = cv2.VideoCapture(0)
    lut = create_identity_lut(N=9)
    lut[..., 0] *= 1.10  # R
    lut[..., 1] *= 1.05  # G
    lut[..., 2] *= 0.95  # B
    lut = np.clip(lut, 0, 1)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        output = apply_3d_lut(frame, lut)
        
        cv2.imshow('Original', frame)
        cv2.imshow('LUT Applied', output[:, :, ::-1])
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()