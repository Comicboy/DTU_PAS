from ultralytics import YOLO
import numpy as np
import pandas as pd
import glob
import cv2
import datetime
import os

from Kalman import kalman_filter

def detect_objects(data):
    '''
    Perform object detection on the input image using a pre-trained YOLO model.
    
    Arguments:
    data: A list of paths to the images

    Returns:
    bbox: A list of detected bounding box objects in all the images
    '''
    bbox = []
    model = YOLO("kitti_yolo_runs/exp1/weights/best.pt")

    tracker = "bytetrack.yaml"
    for image_path in data:
        img = cv2.imread(image_path)
        result = model.track(img, persist = True, tracker = tracker, iou = 0.9, conf = 0.7)

        try:
            bbox.append(result[0].boxes)
        except Exception as e:
            print(f"Error processing result: {e}")
            return None
    
    return bbox

def frame_bb(img, boxes, cls, track, trackChecker = None):
    '''
    Draw bounding boxes and class labels on the input image based on detection results.

    Arguments:
    img: Input image in BGR format
    boxes: Detected bounding boxes
    cls: Class labels for the detected bounding boxes
    track: Track ID for the detected bounding boxes
    trackChecker: Tracks only a certain objects

    Returns:
    imgFrame: Image with drawn bounding boxes and class labels
    '''
    imgFrame = img.copy()
        
    for idx, box in enumerate(boxes):
        coords = box
        x1, y1, x2, y2 = map(int, coords)
        
        color = (int(track[idx] * 10 % 130), int(track[idx] * 70 % 205), int(track[idx] * 160 % 65))
        if trackChecker is None or track[idx] == trackChecker:
            cv2.rectangle(imgFrame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(imgFrame, cls[idx], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return imgFrame

def logData(bbox, kalman0):
    '''
    Log the data required for the tracking video

    Arguments:
    bbox: A list of detected bounding box objects in all the images
    kalman0: A dictionary of the initial values for all the matrices in the kalman filter
    
    Returns:
    df: A data frame for all the necessary data (that can be acquired in 2D)
    as was shown in the projects README file
    '''
    x0 = kalman0["x"]
    u = kalman0["u"]
    P0 = kalman0["P"]
    F = kalman0["F"]
    H = kalman0["H"]
    R = kalman0["R"]

    class_names = {0: 'Car', 1: 'Car', 2: 'Car', 3: 'Pedestrian', 4: 'Pedestrian', 5: 'Cyclist', 6: 'Car', 7: 'Other'}
    log = {"frame": [], "track id": [], "type": [],
        "truncated": [], "occluded": [], "alpha": [],
        "bbox left": [], "bbox top": [], "bbox right": [], "bbox bottom": [],
        "score": []}
    kalmanLogPrev = {"track id": [], "x": [], "x_est": [], "P": [], "Z": []}

    for i, frameBox in enumerate(bbox):
        img = cv2.imread(data[i])
        h, w = img.shape[:2]

        kalmanLog = {"track id": [], "x": [], "x_est": [], "P": [], "Z": []}

        if frameBox is None:
            kalmanLog = kalmanLogPrev.copy()
            for idx, Z in enumerate(kalmanLogPrev["Z"]):
                x, P = kalman_filter(kalmanLogPrev["x_est"][idx], kalmanLogPrev["P"][idx], Z, F, H, R, u)
                kalmanLog["track id"][idx] = kalmanLogPrev["track id"][idx]
                kalmanLog["x"][idx] = kalmanLogPrev["x_est"][idx]
                kalmanLog["x_est"][idx] = x
                kalmanLog["P"][idx] = P
                kalmanLog["Z"][idx] = Z
            continue

        frame = [i] * len(frameBox)
        trackID = [int(ID) for ID in frameBox.id]
        types = [class_names[int(box.cls)] for box in frameBox]
        
        truncate = []
        for box in frameBox:
            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
            x1, y1, x2, y2 = map(int, coords)

            outOfBounds = x1 <= 0 or x2 >= w or y1 <= 0 or y2 >= h
            truncate.append(1 if outOfBounds else 0)

        boxLeft = [int(box[0]) for box in frameBox.xyxy]
        boxTop = [int(box[1]) for box in frameBox.xyxy]
        boxRight = [int(box[2]) for box in frameBox.xyxy]
        boxBottom = [int(box[3]) for box in frameBox.xyxy]
        score = [float(box) for box in frameBox.conf]

        occlude = [0] * len(frameBox)
        alpha = []
        if i > 0:
            for j, currID in enumerate(frameBox.id):
                if currID in log["track id"][-1]:
                    idx = np.where(np.array(log["track id"][-1]) == currID)[0][0]
                    yAxis = abs(boxTop[j] - log["bbox top"][-1][idx])
                    xAxis = abs(boxLeft[j] - log["bbox left"][-1][idx])
                    alphaVal = np.arctan2(yAxis, xAxis)
                    boxIdx = np.where(np.array(kalmanLogPrev["track id"]) == currID)[0][0]

                    alpha.append(alphaVal)
                    kalmanLog["x"].append(kalmanLogPrev["x_est"][boxIdx])
                else:
                    alpha.append(0)
                    kalmanLog["x"].append(x0)

                kalmanLog["track id"].append(int(currID))
                kalmanLog["x_est"].append(x0)
                kalmanLog["P"].append(P0)
                kalmanLog["Z"] = list(np.array([boxLeft, boxTop]).T)

            for j, prevID in enumerate(log["track id"][-1]):
                if prevID not in frameBox.id and log["truncated"][-1][j] == 0:
                    frame.append(i)
                    trackID.append(log["track id"][-1][j])
                    types.append(log["type"][-1][j])

                    boxIdx = np.where(np.array(kalmanLogPrev["track id"]) == prevID)[0][0]
                    predCenter = kalmanLogPrev["x_est"][boxIdx][[0, 3]].reshape(-1,)
                    currCenter = np.array([log["bbox left"][-1][j], log["bbox top"][-1][j]])
                    displace = currCenter - predCenter
                    x1, y1 = log["bbox left"][-1][j] + displace[0], log["bbox top"][-1][j] + displace[1]
                    x2, y2 = log["bbox right"][-1][j] + displace[0], log["bbox bottom"][-1][j] + displace[1]
                    outOfBounds = x1 <= 0 or x2 >= w or y1 <= 0 or y2 >= h

                    truncate.append(1 if outOfBounds else 0)
                    occlude.append(1)

                    boxLeft.append(x1)
                    boxTop.append(y1)
                    boxRight.append(x2)
                    boxBottom.append(y2)
                    score.append(0)

                    yAxis = abs(y1 - log["bbox top"][-1][j])
                    xAxis = abs(x1 - log["bbox left"][-1][j])
                    alphaVal = np.arctan2(yAxis, xAxis)
                    alpha.append(alphaVal)
                    
                    kalmanLog["track id"].append(prevID)
                    kalmanLog["x"].append(kalmanLogPrev["x_est"][boxIdx])
                    kalmanLog["x_est"].append(x0)
                    kalmanLog["P"].append(P0)
                    kalmanLog["Z"].append(np.empty((0, 0)))
        else:
            alpha = [0] * len(frameBox)

            kalmanLog["track id"] = list(frameBox.id)
            kalmanLog["x"] = [x0] * len(frameBox)
            kalmanLog["x_est"] = [x0] * len(frameBox)
            kalmanLog["P"] = [P0] * len(frameBox)
            kalmanLog["Z"] = kalmanLog["Z"] = list(np.array([boxLeft, boxTop]).T)

        log["frame"].append(frame)
        log["track id"].append(trackID)
        log["type"].append(types)
        log["truncated"].append(truncate)
        log["occluded"].append(occlude)
        log["alpha"].append(alpha)
        log["bbox left"].append(boxLeft)
        log["bbox top"].append(boxTop)
        log["bbox right"].append(boxRight)
        log["bbox bottom"].append(boxBottom)
        log["score"].append(score)
        
        for idx, Z in enumerate(kalmanLog["Z"]):
            x, P = kalman_filter(kalmanLog["x"][idx], kalmanLog["P"][idx], Z, F, H, R, u)
            kalmanLog["x_est"][idx] = x
            kalmanLog["P"][idx] = P
        
        kalmanLogPrev = kalmanLog.copy()

    log["frame"] = [item for sublist in log["frame"] for item in sublist]
    log["track id"] = [item for sublist in log["track id"] for item in sublist]
    log["type"] = [item for sublist in log["type"] for item in sublist]
    log["truncated"] = [item for sublist in log["truncated"] for item in sublist]
    log["occluded"] = [item for sublist in log["occluded"] for item in sublist]
    log["alpha"] = [item for sublist in log["alpha"] for item in sublist]
    log["bbox left"] = [item for sublist in log["bbox left"] for item in sublist]
    log["bbox top"] = [item for sublist in log["bbox top"] for item in sublist]
    log["bbox right"] = [item for sublist in log["bbox right"] for item in sublist]
    log["bbox bottom"] = [item for sublist in log["bbox bottom"] for item in sublist]
    log["score"] = [item for sublist in log["score"] for item in sublist]

    df = pd.DataFrame(log)

    return df

def save(height, width, fps = 10, name = "Pikachu"):
    out = cv2.VideoWriter(name + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame in video:
        out.write(frame)
    out.release()

    df.to_csv(name + ".csv", index = False)

if __name__ == "__main__":
    currentPath = "C:/Users/shaia/Documents/Opgaveregning/AutoSys/4. semester/Perception for autonome systemer/Eksamprojekt"
    dataPath = currentPath + "/34759_final_project_rect/seq_01/image_02"
    dataPNG = dataPath + "/data"
    dataTime = dataPath + "/timestamps.txt"

    data = glob.glob(f"{dataPNG}/*.png")
    timeStamp = open(dataTime)
    timeData = timeStamp.readlines()
    timeStamp.close()
    time1 = datetime.datetime.strptime(timeData[0][:-4], "%Y-%m-%d %H:%M:%S.%f")
    time2 = datetime.datetime.strptime(timeData[1][:-4], "%Y-%m-%d %H:%M:%S.%f")
    deltaT = (time2 - time1).total_seconds()

    x0 = np.zeros((6, 1))
    u = np.zeros((6, 1))
    P0 = 1000 * np.eye(x0.shape[0])
    F = np.array([[1, deltaT, 0.5 * deltaT ** 2, 0, 0, 0],
                [0, deltaT, deltaT ** 2, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, deltaT, 0.5 * deltaT ** 2],
                [0, 0, 0, 0, deltaT, deltaT ** 2],
                [0, 0, 0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]])
    R = 1 * np.eye(np.sum(H == 1))
    kalman0 = {"x": x0, "u": u, "P": P0, "F": F, "H": H, "R": R}

    bbox = detect_objects(data)
    df = logData(bbox, kalman0)

    video = []
    for i in range(max(df["frame"])):
        img = cv2.imread(data[i])

        if i not in df["frame"]:
            video.append(img)
            continue
        
        subData = df.loc[df["frame"] == i]
        coords = np.array([subData["bbox left"], subData["bbox top"], subData["bbox right"], subData["bbox bottom"]]).T
        types = list(subData["type"])
        trackID = list(subData["track id"])

        imgFrame = frame_bb(img, coords, types, trackID)
        video.append(imgFrame)

    for image in video:
        cv2.namedWindow('Frames', cv2.WINDOW_NORMAL)
        cv2.imshow('Frames', image)
        cv2.waitKey(int(deltaT * 1e+3))
    cv2.destroyAllWindows()

    h, w = video[0].shape[:2]
    save(h, w, 1 / deltaT, "Result")