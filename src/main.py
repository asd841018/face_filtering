import time
import yaml
import cv2
import numpy as np

from model import FaceModel
from utils.smooth import smooth_face
from utils.adjust_color import adjust_white_ycrcb
from utils.vingtage import apply_curve_bgr, build_fade_curve, vignette

def load_configs():
    """
    Loads the project configurations.

    Returns
    -------
    configs : dict
        A dictionary containing the configs
    """
    with open('D:/work/face_filtering/src/configs/configs.yaml', 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def main():
    configs = load_configs()
    method = "vintage"
    cap = cv2.VideoCapture(0)
    width = 1280
    height = 720

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Load Face Model
    face_model = FaceModel(configs)
    
    curve = build_fade_curve(contrast=0.12, lift=14, gamma=1.05)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detected_img = frame.copy()
        s = time.time()
        result = face_model.detect_faces(detected_img)
        bbox = result['bbox']
        output_img, roi_img, full_mask, smoothed_roi = smooth_face(configs, detected_img, bbox)
        if method == "color":
            output_img, _ = adjust_white_ycrcb(output_img, strength=0.35, brighten=6, mode="cold")
        else:
            output_img = apply_curve_bgr(output_img, curve)
            output_img = vignette(output_img, strength=0.35)
            
        print(f"FPS: {1/(time.time() - s):.2f}")
        cv2.imshow('Smoothed Face', output_img)
        cv2.imshow('Full Mask', full_mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()