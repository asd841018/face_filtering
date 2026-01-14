import cv2
import numpy as np
from scipy.interpolate import Rbf


def get_face_control_points_106(landmarks: np.ndarray):
    """
    Get face control points for deformation from InsightFace 106 landmarks
    
    Args:
        landmarks: (106, 2) numpy array, each row is (x, y) pixel coordinates
    
    Returns:
        src_points: Control point coordinates
        indices: Control point indices
        left_cheek_indices: Left cheek indices
        right_cheek_indices: Right cheek indices
        chin_indices: Chin indices
    """
    # Right cheek control points
    RIGHT_CHEEK = [1, 9, 10, 11, 12, 13, 14, 15, 16]
    # Left cheek control points
    LEFT_CHEEK = [17, 25, 26, 27, 28, 29, 30, 31, 32]
    # Chin control points
    CHIN = [2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18]
    # Nose (fixed)
    NOSE = [73, 74, 76, 77, 78, 79, 80, 85, 84, 83, 82, 86]
    # Eyes (fixed)
    RIGHT_EYE = [35, 36, 33, 37, 39, 42, 40, 41, 38]
    LEFT_EYE = [93, 96, 94, 95, 89, 90, 87, 91, 88]
    # Mouth (fixed)
    MOUTH = [71, 63, 64, 52, 55, 56, 53, 59, 58, 61, 68, 67, 62, 66, 65, 54, 60, 57, 69, 70]
    # Eyebrows (fixed)
    RIGHT_EYEBROWS = [43, 48, 49, 51, 50, 46, 47, 45, 44]
    LEFT_EYEBROWS = [101, 105, 104, 103, 102, 97, 98, 99, 100]
    # Between eyes (fixed)
    BETWEEN_EYES = [75, 72, 81]
    
    # Combine all control points
    all_indices = (LEFT_CHEEK + RIGHT_CHEEK + CHIN + 
                   NOSE + LEFT_EYE + RIGHT_EYE + MOUTH + RIGHT_EYEBROWS + LEFT_EYEBROWS + BETWEEN_EYES)
    
    # Remove duplicates
    all_indices = list(set(all_indices))
    all_indices.sort()
    
    src_points = landmarks[all_indices].copy()
    
    return src_points, all_indices, LEFT_CHEEK, RIGHT_CHEEK, CHIN


def apply_slim_deformation_106(src_points: np.ndarray, 
                                indices: list,
                                left_cheek_indices: list,
                                right_cheek_indices: list, 
                                chin_indices: list,
                                landmarks: np.ndarray,
                                cheek_strength: float = 0.15,
                                chin_strength: float = 0.10):
    """
    Apply slim face deformation
    
    Args:
        src_points: Control point coordinates
        indices: Control point indices
        left_cheek_indices: Left cheek indices
        right_cheek_indices: Right cheek indices
        chin_indices: Chin indices
        landmarks: Complete 106 landmarks
        cheek_strength: Cheek inward offset strength (0~1)
        chin_strength: Chin upward contraction strength (0~1)
    
    Returns:
        dst_points: Deformed control point coordinates
    """
    dst_points = src_points.copy()
    
    # Get nose tip as center reference point (landmark 86)
    nose_center = landmarks[86]
    nose_x, nose_y = nose_center[0], nose_center[1]
    
    # Get lowest chin point (landmark 0 is the lowest chin center point)
    chin_bottom = landmarks[0]
    chin_bottom_y = chin_bottom[1]
    
    # Calculate face width for normalization
    face_width = np.abs(landmarks[1][0] - landmarks[17][0])  # Right temple to left temple
    face_height = np.abs(landmarks[0][1] - landmarks[49][1])  # Chin to forehead
    
    # Offset left cheek points (toward nose)
    for idx in left_cheek_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            
            # Calculate vector toward nose
            dx = nose_x - x
            dy = nose_y - y
            
            # Adjust strength based on distance from nose (farther distance = stronger effect)
            distance = np.sqrt(dx**2 + dy**2)
            distance_factor = distance / face_width if face_width > 0 else 0
            
            # Adjust strength based on y coordinate (middle cheek area has stronger effect)
            # Middle cheek position is approximately near the nose
            y_factor = 1.0 - abs(y - nose_y) / (face_height * 0.5) if face_height > 0 else 1.0
            y_factor = max(0.3, min(1.0, y_factor))
            
            strength = cheek_strength * (0.5 + 0.5 * distance_factor) * y_factor
            
            dst_points[orig_idx][0] = x + dx * strength
            dst_points[orig_idx][1] = y + dy * strength * 0.3  # Smaller Y direction offset
    
    # Offset right cheek points (toward nose)
    for idx in right_cheek_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            
            dx = nose_x - x
            dy = nose_y - y
            
            distance = np.sqrt(dx**2 + dy**2)
            distance_factor = distance / face_width if face_width > 0 else 0
            
            y_factor = 1.0 - abs(y - nose_y) / (face_height * 0.5) if face_height > 0 else 1.0
            y_factor = max(0.3, min(1.0, y_factor))
            
            strength = cheek_strength * (0.5 + 0.5 * distance_factor) * y_factor
            
            dst_points[orig_idx][0] = x + dx * strength
            dst_points[orig_idx][1] = y + dy * strength * 0.3
    
    # Contract chin points upward
    for idx in chin_indices:
        if idx in indices:
            orig_idx = indices.index(idx)
            x, y = src_points[orig_idx]
            
            # Offset upward and toward center
            dy = chin_bottom_y - y
            dx = nose_x - x
            
            # Adjust strength based on distance from lowest chin point
            distance_factor = abs(dy) / face_height if face_height > 0 else 0
            strength = chin_strength * (1.0 - distance_factor * 0.5)
            
            # Contract chin upward
            dst_points[orig_idx][0] = x + dx * strength * 0.2
            dst_points[orig_idx][1] = y - abs(y - nose_y) * strength * 0.15
    
    return dst_points


def warp_face_rbf(img: np.ndarray, 
                  src_points: np.ndarray, 
                  dst_points: np.ndarray,
                  grid_resolution: int = 50):
    """
    Fast grid deformation using RBF (Radial Basis Function)
    
    Args:
        img: Input image
        src_points: Original control points
        dst_points: Target control points
        grid_resolution: Grid resolution (smaller = faster)
    
    Returns:
        warped: Warped image
    """
    h, w = img.shape[:2]
    
    # Add boundary control points to fix image edges
    margin = max(40, min(w, h) // 10)
    border_points = []
    
    # Four corners
    border_points.extend([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    
    # Border points
    for x in range(margin, w-margin, margin):
        border_points.extend([[x, 0], [x, h-1]])
    for y in range(margin, h-margin, margin):
        border_points.extend([[0, y], [w-1, y]])
    
    border_points = np.array(border_points, dtype=np.float32)
    
    # Combine face control points and border points
    all_src_points = np.vstack([src_points, border_points])
    all_dst_points = np.vstack([dst_points, border_points])
    
    # Build RBF interpolation (thin_plate function)
    try:
        rbf_x = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 0],
                    function='thin_plate', smooth=0)
        rbf_y = Rbf(all_dst_points[:, 0], all_dst_points[:, 1], all_src_points[:, 1],
                    function='thin_plate', smooth=0)
    except Exception as e:
        # If RBF fails, return original image
        return img
    
    # Use downsampled grid to accelerate computation
    grid_w = max(1, w // grid_resolution)
    grid_h = max(1, h // grid_resolution)
    
    grid_x_small = np.linspace(0, w-1, grid_w)
    grid_y_small = np.linspace(0, h-1, grid_h)
    grid_x_small, grid_y_small = np.meshgrid(grid_x_small, grid_y_small)
    
    # Calculate deformation on low resolution grid
    map_x_small = rbf_x(grid_x_small, grid_y_small).astype(np.float32)
    map_y_small = rbf_y(grid_x_small, grid_y_small).astype(np.float32)
    
    # Interpolate upscale to original resolution
    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Execute remapping
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    
    return warped


def reshape_face(img: np.ndarray,
                 landmarks: np.ndarray,
                 cheek_strength: float = 0.15,
                 chin_strength: float = 0.10,
                 grid_resolution: int = 50) -> np.ndarray:
    """
    Main function: Apply face slimming effect
    
    Args:
        img: Input image (BGR or RGB)
        landmarks: InsightFace 106 landmarks, shape is (106, 2)
        cheek_strength: Cheek contraction strength (0~1), larger = slimmer cheeks
        chin_strength: Chin contraction strength (0~1), larger = sharper chin
        grid_resolution: Grid resolution (20-80), smaller = faster but lower precision
    
    Returns:
        reshaped: Face-slimmed image
    """
    if landmarks is None or len(landmarks) != 106:
        return img
    
    # Ensure landmarks are float32
    landmarks = landmarks.astype(np.float32)
    
    # 1. Get control points
    src_points, indices, left_cheek, right_cheek, chin = get_face_control_points_106(landmarks)
    
    # 2. Calculate deformed target points
    dst_points = apply_slim_deformation_106(
        src_points, indices, left_cheek, right_cheek, chin,
        landmarks, cheek_strength, chin_strength
    )
    
    # 3. Apply RBF deformation
    reshaped = warp_face_rbf(img, src_points, dst_points, grid_resolution)
    
    return reshaped


def reshape_faces(img: np.ndarray,
                  faces: list,
                  cheek_strength: float = 0.15,
                  chin_strength: float = 0.10,
                  grid_resolution: int = 50) -> np.ndarray:
    """
    Apply face slimming to all faces in the image
    
    Args:
        img: Input image (BGR or RGB)
        faces: List of faces detected by InsightFace
        cheek_strength: Cheek contraction strength (0~1)
        chin_strength: Chin contraction strength (0~1)
        grid_resolution: Grid resolution
    
    Returns:
        reshaped: Face-slimmed image
    """
    result = img.copy()
    
    for face in faces:
        landmarks = face.get("landmark_2d_106")
        if landmarks is not None and len(landmarks) == 106:
            result = reshape_face(
                result, landmarks,
                cheek_strength, chin_strength, grid_resolution
            )
    
    return result


# Test code
if __name__ == "__main__":
    import time
    from insightface.app import FaceAnalysis
    
    # Initialize InsightFace
    print("Initializing InsightFace...")
    model = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=["detection", "landmark_2d_106"],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    model.prepare(ctx_id=0, det_size=(640, 640))
    
    # Open video
    cap = cv2.VideoCapture(0)
    
    # Parameter settings
    cheek_strength = 0.15  # Cheek contraction strength
    chin_strength = 0.10   # Chin contraction strength
    grid_resolution = 50   # Grid resolution
    
    print("\n=== Face Reshape with InsightFace 106 Landmarks ===")
    print(f"Cheek strength: {cheek_strength}")
    print(f"Chin strength: {chin_strength}")
    print(f"Grid resolution: {grid_resolution}")
    print("\nControls:")
    print("  'q' - Quit")
    print("  '+/-' - Adjust cheek strength")
    print("  '[/]' - Adjust chin strength")
    print("  'd' - Toggle dual view")
    
    show_dual_view = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop playback
            continue
        
        # Downscale for faster processing
        frame_small = cv2.resize(frame, (640, 360))
        
        s = time.time()
        
        # Detect faces
        faces = model.get(frame_small)
        
        # Face slimming
        if len(faces) > 0:
            result = reshape_faces(
                frame_small, faces,
                cheek_strength, chin_strength, grid_resolution
            )
        else:
            result = frame_small.copy()
        
        fps = 1.0 / (time.time() - s)
        
        # Display FPS
        cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result, f"Cheek: {cheek_strength:.2f} Chin: {chin_strength:.2f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if show_dual_view:
            # Display original
            original = frame_small.copy()
            cv2.putText(original, "Original", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display side by side
            combined = np.hstack([original, result])
            cv2.imshow("Face Reshape (Original | Reshaped)", combined)
        else:
            cv2.imshow("Face Reshape", result)
        
        # Keyboard control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            cheek_strength = min(1.0, cheek_strength + 0.05)
            print(f"Cheek strength: {cheek_strength:.2f}")
        elif key == ord('-') or key == ord('_'):
            cheek_strength = max(0.0, cheek_strength - 0.05)
            print(f"Cheek strength: {cheek_strength:.2f}")
        elif key == ord('['):
            chin_strength = max(0.0, chin_strength - 0.05)
            print(f"Chin strength: {chin_strength:.2f}")
        elif key == ord(']'):
            chin_strength = min(1.0, chin_strength + 0.05)
            print(f"Chin strength: {chin_strength:.2f}")
        elif key == ord('d'):
            show_dual_view = not show_dual_view
            cv2.destroyAllWindows()
    
    cap.release()
    cv2.destroyAllWindows()
