import cv2
import numpy as np
from ultralytics import YOLO

###############################################
# 1. DEFINE CAMERA PARAMETERS
###############################################
# (Kept identical to your original provided code)

# Camera 2 intrinsics
K_02 = np.array([
    [9.569475e+02, 0.0,          6.939767e+02],
    [0.0,          9.522352e+02, 2.386081e+02],
    [0.0,          0.0,          1.0]
])
D_02 = np.array([-3.750956e-01, 2.076838e-01, 4.348525e-04, 1.603162e-03, -7.469243e-02])

R_02 = np.array([
    [ 9.999838e-01, -5.012736e-03, -2.710741e-03],
    [ 5.002007e-03,  9.999797e-01, -3.950381e-03],
    [ 2.730489e-03,  3.936758e-03,  9.999885e-01]
])
T_02 = np.array([5.989688e-02, -1.367835e-03, 4.637624e-03]).reshape(3,1)

# Camera 3 intrinsics
K_03 = np.array([
    [9.011007e+02, 0.0,          6.982947e+02],
    [0.0,          8.970639e+02, 2.377447e+02],
    [0.0,          0.0,          1.0]
])
D_03 = np.array([-3.686011e-01, 1.908666e-01, -5.689518e-04, 3.332341e-04, -6.302873e-02])

R_03 = np.array([
    [ 9.995054e-01,  1.665288e-02, -2.667675e-02],
    [-1.671777e-02,  9.998578e-01, -2.211228e-03],
    [ 2.663614e-02,  2.656110e-03,  9.996417e-01]
])
T_03 = np.array([-4.756270e-01, 5.296617e-03, -5.437198e-03]).reshape(3,1)

K_left = K_02        
K_right = K_03
D_left = D_02
D_right = D_03

R = R_03 @ R_02.T
T = T_03 - R @ T_02

# Stereo baseline (in meters)
baseline = abs(T[0,0])

###############################################
# 3. RECTIFICATION
###############################################
image_size = (1392, 512)   # KITTI camera2/3 original resolution

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K_left, D_left, K_right, D_right,
    image_size, R, T, alpha=0
)

# Compute undistort/rectify maps
left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    K_left, D_left, R1, P1, image_size, cv2.CV_32FC1
)
right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    K_right, D_right, R2, P2, image_size, cv2.CV_32FC1
)


###############################################
# 4. SGBM MATCHING (UPDATED)
###############################################
# Added speckle filtering and mode 3WAY for better accuracy/robustness

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,      
    blockSize=5,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,
    disp12MaxDiff=1,         # Consistency check
    uniquenessRatio=10,      # Filter out pixels if best match isn't significantly better than 2nd best
    speckleWindowSize=100,   # Remove small "islands" of noise (white dots)
    speckleRange=32,         # Threshold for speckle detection
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY # Better accuracy
)

###############################################
# 5. COMPUTE DISPARITY MAP AND DO 3D REPROJECTION
###############################################

def compute_disparity(left_img, right_img):
    left_rect = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Note: result is 16-bit signed, we divide by 16.0 to get real disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disparity, Q, left_rect

# Convert disparity -> 3D coordinates
def disparity_to_points_3d(disparity, Q):
    return cv2.reprojectImageTo3D(disparity, Q)


###############################################
# 6. OBTAIN DEPTH INFORMATION FROM YOLO BBOX (UPDATED)
###############################################
# Now uses Median depth of the ROI, rather than a single center pixel

def get_depth_from_yolo_box(box, points_3d, disparity_map):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

    # Ensure coordinates are within image bounds
    h, w = points_3d.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    # Safety check: if box is invalid or 0 size
    if x1 >= x2 or y1 >= y2:
        return None

    # Crop the 3D ROI and the Disparity ROI
    roi_3d = points_3d[y1:y2, x1:x2]
    roi_disp = disparity_map[y1:y2, x1:x2]

    # Filter: valid disparity points (disparity > 0)
    # SGBM usually returns 0 or -1 for invalid matches
    mask = roi_disp > 0  
    
    # Extract Z values where disparity was valid
    valid_depths = roi_3d[mask, 2] 

    # Further filter: remove Crazy outliers (e.g. negative Z or Z > 150m)
    # This helps when reprojectImageTo3D creates massive numbers for disp~0
    valid_depths = valid_depths[(valid_depths > 0.5) & (valid_depths < 150.0)]

    if len(valid_depths) == 0:
        return None # No valid depth info found in this box

    # Use Median to ignore outliers/reflections
    Z = np.median(valid_depths)
    
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return {
        "pixel_center": (cx, cy),
        "Z": float(Z),
        "class": int(box.cls[0]),
        "confidence": float(box.conf[0])
    }

###############################################
# 7. FULL PIPELINE (UPDATED)
###############################################

def process_stereo_images(left_img_path, right_img_path, yolo_model_path="yolo.pt"):
    # Load images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    # Compute disparity + rectified left
    disparity, Q, left_rect = compute_disparity(left_img, right_img)

    # Convert disparity -> 3D map
    points_3d = disparity_to_points_3d(disparity, Q)

    # Run YOLO on the (rectified) left image
    model = YOLO(yolo_model_path)
    
    # Optional: Increase inference size for small objects (KITTI is wide)
    # results = model(left_rect, imgsz=1280)[0] 
    results = model(left_rect)[0]

    depth_results = []

    for box in results.boxes:
        # Pass disparity map to helper now
        depth_info = get_depth_from_yolo_box(box, points_3d, disparity)
        
        if depth_info is not None:
            depth_results.append(depth_info)

            # Draw the bounding box and depth on image
            cx, cy = depth_info["pixel_center"]
            Z = depth_info["Z"]
            cls_id = depth_info["class"]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # Color code: Green usually, but Red if very close (< 10m)
            color = (0, 0, 255) if Z < 10.0 else (0, 255, 0)
            
            cv2.rectangle(left_rect, (x1,y1), (x2,y2), color, 2)

            label_text = f"C:{cls_id} Z={Z:.2f}m"
            cv2.putText(left_rect,
                        label_text,
                        (x1, y1 - 10), # Put text above box
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        else:
            # Draw box but indicate unknown depth
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(left_rect, (x1,y1), (x2,y2), (128,128,128), 2)
            cv2.putText(left_rect, "Z=?", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 2)

    return left_rect, disparity, depth_results

# Example usage

if __name__ == "__main__":
    left_image = "left.png" 
    right_image = "right.png"
    yolo_weights = "yolov8n.pt" # Or your trained "runs/.../weights/best.pt"

    annotated_img, disparity_map, depths = process_stereo_images(
        left_image, right_image, yolo_weights
    )

    # Print all detected object depths:
    for d in depths:
        print(f"Class {d['class']} | Depth: {d['Z']:.2f} m | Center: {d['pixel_center']}")

    cv2.imshow("Detections with Depth", annotated_img)
    
    # Normalize disparity for visualization
    disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("Disparity", disp_vis)
    
    cv2.waitKey(0)