import cv2
import numpy as np
from ultralytics import YOLO
import time
import glob
import os

###############################################
# 1. DEFINE CAMERA PARAMETERS
###############################################
# (Kept identical to previous versions)

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
# 3. RECTIFICATION SETUP (RUNS ONCE)
###############################################
image_size = (1392, 512)   

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
# 4. SGBM MATCHING
###############################################

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,      
    blockSize=5,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5,
    disp12MaxDiff=1,         
    uniquenessRatio=10,      
    speckleWindowSize=100,   
    speckleRange=32,         
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

###############################################
# 6. HELPER FUNCTIONS
###############################################

def compute_disparity(left_img, right_img):
    # Remap images using the pre-calculated maps
    left_rect = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    # Compute disparity
    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disparity, Q, left_rect

def get_depth_from_yolo_box(box, points_3d, disparity_map):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    h, w = points_3d.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    if x1 >= x2 or y1 >= y2: return None

    roi_3d = points_3d[y1:y2, x1:x2]
    roi_disp = disparity_map[y1:y2, x1:x2]

    mask = roi_disp > 0  
    valid_depths = roi_3d[mask, 2] 
    valid_depths = valid_depths[(valid_depths > 0.5) & (valid_depths < 150.0)]

    if len(valid_depths) == 0: return None

    Z = np.median(valid_depths)
    
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    return {
        "pixel_center": (cx, cy),
        "Z": float(Z),
        "class": int(box.cls[0]),
        "confidence": float(box.conf[0])
    }

def process_frame(frameL, frameR, yolo_model):
    disparity, Q, left_rect = compute_disparity(frameL, frameR)
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    results = yolo_model(left_rect, verbose=False)[0]

    for box in results.boxes:
        depth_info = get_depth_from_yolo_box(box, points_3d, disparity)
        
        if depth_info is not None:
            cx, cy = depth_info["pixel_center"]
            Z = depth_info["Z"]
            cls_id = depth_info["class"]

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color = (0, 0, 255) if Z < 10.0 else (0, 255, 0)
            
            cv2.rectangle(left_rect, (x1,y1), (x2,y2), color, 2)
            label_text = f"C:{cls_id} Z={Z:.2f}m"
            cv2.putText(left_rect, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(left_rect, (x1,y1), (x2,y2), (128,128,128), 2)
            cv2.putText(left_rect, "Z=?", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 2)

    return left_rect, disparity

###############################################
# 7. MAIN FOLDER LOOP (MODIFIED)
###############################################

if __name__ == "__main__":
    left_folder = "data/image_02/data/"   
    right_folder = "data/image_03/data/"  
    yolo_path = "kitti_yolo_runs/exp1/weights/best.pt"

    print("Loading YOLO model...")
    yolo_model = YOLO(yolo_path)
    
    left_files = sorted(glob.glob(os.path.join(left_folder, "*.png")))
    right_files = sorted(glob.glob(os.path.join(right_folder, "*.png")))

    if not left_files or not right_files:
        print("Error: No images found.")
        exit()

    num_frames = min(len(left_files), len(right_files))
    print(f"Starting processing for {num_frames} frames...")

    # Flag to ensure we only save once
    has_saved_disparity = False

    for i in range(num_frames):
        start_time = time.time()

        frameL = cv2.imread(left_files[i])
        frameR = cv2.imread(right_files[i])

        if frameL is None or frameR is None: continue

        # --- PROCESS FRAME ---
        annotated_img, disparity_map = process_frame(frameL, frameR, yolo_model)

        # --- SAVE DISPARITY MAP (ONCE) ---
        if not has_saved_disparity:
            print(f"Saving disparity map for Frame {i}...")
            
            # Normalize and Colorize
            disp_vis = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            
            cv2.imwrite("single_disparity_sample.png", disp_color)
            has_saved_disparity = True
            print("Saved 'single_disparity_sample.png' successfully.")

        # --- VISUALIZATION ---
        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0
        cv2.putText(annotated_img, f"FPS: {fps:.1f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Stereo YOLO Depth", annotated_img)
        
        # We still show the live disparity in the window, even if not saving
        disp_live = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imshow("Disparity Live", disp_live)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("Processing complete.")