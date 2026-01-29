import cv2
import numpy as np

def skin_mask_from_ycrcb(ycrcb):
    return cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))

def get_roi(detected_img, bbox):
    """
    Crop detected image to size of detection

    Parameters
    ----------
    detected_img : np.array [H,W,3]
        BGR image
    bbox : list or np.array
        Bounding box coordinates [x1, y1, x2, y2]
    """
    return detected_img[int(bbox[1]):int(bbox[3]), 
                        int(bbox[0]):int(bbox[2])]


def smooth_face(cfg, detected_img, bbox):
    """
    Smooth faces in an image using bilateral filtering.

    Parameters
    ----------
    cfg : dict
        Dictionary of configurations
    box_face : np.array [H,W,3]
        BGR image
    bboxes : list
        List of detected bounding boxes

    Returns
    -------
    detected_img : np.array [H,W,3]
        BGR image with face detections
    roi : np.array [H,W,3]
        BGR image
    full_mask : np.array [H,W,3]
        BGR image
    full_img : np.array [H,W,3]
        BGR image
    """
    # Get Region of Interest
    roi_img = get_roi(detected_img, bbox)
    # Copy ROI
    temp_img = roi_img.copy()
    ycrcb_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2YCrCb)
    ycrcb_mask = skin_mask_from_ycrcb(ycrcb_img)
    # Make a 3 channel mask
    full_mask = cv2.merge((ycrcb_mask, ycrcb_mask, ycrcb_mask))
    # Apply blur on the created image
    blurred_img = cv2.bilateralFilter(roi_img, 
                                        cfg['filter']['diameter'], 
                                        cfg['filter']['sigma_1'], 
                                        cfg['filter']['sigma_2'])
    # Apply mask to image
    masked_img = cv2.bitwise_and(blurred_img, full_mask)
    # Invert mask
    inverted_mask = cv2.bitwise_not(full_mask)
    # Created anti-mask
    masked_img2 = cv2.bitwise_and(temp_img, inverted_mask)
    # Add the masked images together
    smoothed_roi = cv2.add(masked_img2, masked_img)
    # Init smoothed image
    output_img = detected_img.copy()
    # Replace ROI on full image with blurred ROI
    output_img[int(bbox[1]):int(bbox[3]), 
               int(bbox[0]):int(bbox[2])] = smoothed_roi
    return output_img, roi_img, full_mask, smoothed_roi