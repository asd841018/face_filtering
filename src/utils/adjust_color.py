import cv2
import numpy as np

def adjust_white_ycrcb(bgr, strength=0.35, brighten=6, mode=None):
    """
    使用 cv2.add/cv2.subtract 取代 numpy 運算（自動處理溢位）
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    cb_delta = int(25 * strength)
    cr_delta = int(22 * strength)

    if mode == "warm":
        Cb2 = cv2.subtract(Cb, cb_delta)
        Cr2 = cv2.add(Cr, cr_delta)
    elif mode == "cold":
        Cb2 = cv2.add(Cb, cb_delta)
        Cr2 = cv2.subtract(Cr, cr_delta)
    else:
        Cb2, Cr2 = Cb, Cr

    # cv2.add 自動 clamp 到 0~255
    Y2 = cv2.add(Y, brighten)

    out = cv2.merge([Y2, Cr2, Cb2])
    return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR), ycrcb