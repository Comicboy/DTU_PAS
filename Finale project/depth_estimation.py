import cv2
import numpy as np
from ultralytics import YOLO

###############################################
# 1. DEFINE CAMERA PARAMETERS
###############################################
# Here camera 2 is used as stereo left and camera 3 is used as stereo right

# Define camera related parameters based on the calibration file
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
# 4. SGBM MATCHING
###############################################

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5,
    P1=8 * 3 * 5 * 5,
    P2=32 * 3 * 5 * 5
)

###############################################
# 5. COMPUTE DISPARITY MAP AND DO 3D REPROJECTION
###############################################

def compute_disparity(left_img, right_img):
    left_rect = cv2.remap(left_img, left_map_x, left_map_y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right_img, right_map_x, right_map_y, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    return disparity, Q, left_rect

# Convert disparity → 3D coordinates

def disparity_to_points_3d(disparity, Q):
    return cv2.reprojectImageTo3D(disparity, Q)


###############################################
# 6. OBTAIN DEPTH INFORMATION FROM YOLO BBOX
###############################################

def get_depth_from_yolo_box(box, points_3d):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    X, Y, Z = points_3d[cy, cx]

    return {
        "pixel_center": (cx, cy),
        "X": float(X),
        "Y": float(Y),
        "Z": float(Z),
        "class": int(box.cls[0]),
        "confidence": float(box.conf[0])
    }

###############################################
# 7. FULL PIPELINE
###############################################

def process_stereo_images(left_img_path, right_img_path, yolo_model_path="yolo.pt"):
    # Load images
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)

    # Compute disparity + rectified left
    disparity, Q, left_rect = compute_disparity(left_img, right_img)

    # Convert disparity → 3D map
    points_3d = disparity_to_points_3d(disparity, Q)

    # Run YOLO on the (rectified) left image
    model = YOLO(yolo_model_path)
    results = model(left_rect)[0]

    depth_results = []

    for box in results.boxes:
        depth_info = get_depth_from_yolo_box(box, points_3d)
        depth_results.append(depth_info)

        # Draw the bounding box and depth on image
        cx, cy = depth_info["pixel_center"]
        Z = depth_info["Z"]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(left_rect, (x1,y1), (x2,y2), (0,255,0), 2)

        cv2.putText(left_rect,
                    f"Z={Z:.2f} m",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

    return left_rect, disparity, depth_results

# Example usage

if __name__ == "__main__":
    left_image = "left.png"
    right_image = "right.png"
    yolo_weights = "yolov8n.pt"

    annotated_img, disparity_map, depths = process_stereo_images(
        left_image, right_image, yolo_weights
    )

    # Print all detected object depths:
    for d in depths:
        print(f"Class {d['class']} | Depth: {d['Z']:.2f} m | Center: {d['pixel_center']}")

    cv2.imshow("Detections with Depth", annotated_img)
    cv2.imshow("Disparity", disparity_map / np.max(disparity_map))
    cv2.waitKey(0)
