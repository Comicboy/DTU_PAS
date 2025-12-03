from ultralytics import YOLO
import numpy as np
import pandas as pd
import glob
import cv2
import datetime

from Kalman import kalman_filter

# Change this to load the data path
currentPath = "C:/Users/shaia/Documents/Opgaveregning/AutoSys/4. semester/Perception for autonome systemer/Eksamprojekt/"

def detect_objects(data):
    '''
    Perform object detection on the input image using a pre-trained YOLO model.
    
    Arguments:
    data: A list of paths to the images

    Returns:
    bbox: A list of detected bounding box objects in all the images
    '''
    bbox = []
    model = YOLO("kitti_yolo_runs/exp2/weights/best.pt")

    tracker = "bytetrack_modified.yaml"
    for image_path in data:
        img = cv2.imread(image_path)
        result = model.track(img, persist = True, tracker = tracker, iou = 0.7, conf = 0.5)

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
    boxes: A list of detected bounding boxes in the mage
    cls: A list of class labels for the detected bounding boxes
    track: A list of track ID for the detected bounding boxes
    trackChecker: An integer for only tracking a certain object

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
    Log the data required for the tracking video.

    Arguments:
    bbox: A list of detected bounding box objects in all the images
    kalman0: A dictionary of the initial values for all the matrices in the kalman filter
    
    Returns:
    df: A data frame for all the necessary data (that can be acquired in 2D)
    as was shown in the project description's README file
    '''
    x0 = kalman0["x"]
    u = kalman0["u"]
    P0car = kalman0["Pcar"]
    P0ped = kalman0["Pped"]
    P0cyc = kalman0["Pcyc"]
    F = kalman0["F"]
    H = kalman0["H"]
    R = kalman0["R"]

    class_names = {0: 'Car', 1: 'Car', 2: 'Car', 3: 'Pedestrian', 4: 'Pedestrian', 5: 'Cyclist', 6: 'Car', 7: 'Other'}
    log = {"frame": [], "track id": [], "type": [],
        "truncated": [], "occluded": [], "alpha": [],
        "bbox left": [], "bbox top": [], "bbox right": [], "bbox bottom": [],
        "score": []}
    kalmanLogPrev = {"track id": [], "x": [], "P": [], "Z": [], "time": []}

    for i, frameBox in enumerate(bbox):
        img = cv2.imread(data[i])
        h, w = img.shape[:2]

        kalmanLog = {"track id": [], "x": [], "P": [], "Z": [], "time": []}

        if frameBox.id is None:
            kalmanLog = kalmanLogPrev.copy()
            for idx, Z in enumerate(kalmanLogPrev["Z"]):
                x, P = kalman_filter(kalmanLogPrev["x"][idx], kalmanLogPrev["P"][idx], Z, F, H, R, u)
                kalmanLog["track id"][idx] = kalmanLogPrev["track id"][idx]
                kalmanLog["x"][idx] = x
                kalmanLog["P"][idx] = P
                kalmanLog["Z"][idx] = Z
            
            kalmanLogPrev = kalmanLog.copy()
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
            for j, currID in enumerate(trackID):
                kalmanLog["track id"].append(int(currID))
                kalmanLog["Z"] = list(np.array([boxLeft, boxTop, boxRight, boxBottom]).T)

                if currID in log["track id"][-1]:
                    yAxis = boxTop[j] - log["bbox top"][-1][idx]
                    xAxis = boxLeft[j] - log["bbox left"][-1][idx]
                    alphaVal = np.arctan2(yAxis, xAxis)
                    boxIdx = np.where(np.array(kalmanLogPrev["track id"]) == currID)[0][0]

                    alpha.append(alphaVal)
                    kalmanLog["P"].append(kalmanLogPrev["P"][boxIdx])
                    kalmanLog["x"].append(kalmanLogPrev["x"][boxIdx])
                    kalmanLog["time"].append(kalmanLogPrev["time"][boxIdx] + 1)
                else:
                    alpha.append(0)
                    currentType = types[j]
                    kalmanLog["P"].append(P0ped if currentType == "Pedestrian" else P0car if currentType == "Car" else P0cyc)
                    kalmanLog["x"].append(x0)
                    kalmanLog["time"].append(0)

            for j, prevID in enumerate(log["track id"][-1]):
                boxIdx = np.where(np.array(kalmanLogPrev["track id"]) == prevID)[0][0]
                if prevID not in trackID and log["truncated"][-1][j] == 0 and kalmanLogPrev["time"][boxIdx] > 10:
                    x1, y1, x2, y2 = kalmanLogPrev["x"][boxIdx][[0, 3, 6, 9]].reshape(-1,)
                    outOfBounds = x1 <= 0 or x2 >= w or y1 <= 0 or y2 >= h
                    yAxis = y1 - log["bbox top"][-1][j]
                    xAxis = x1 - log["bbox left"][-1][j]
                    alphaVal = np.arctan2(yAxis, xAxis)

                    frame.append(i)
                    trackID.append(log["track id"][-1][j])
                    types.append(log["type"][-1][j])
                    truncate.append(1 if outOfBounds else 0)
                    occlude.append(1)
                    alpha.append(alphaVal)
                    boxLeft.append(x1)
                    boxTop.append(y1)
                    boxRight.append(x2)
                    boxBottom.append(y2)
                    score.append(0)
                    
                    kalmanLog["track id"].append(prevID)
                    kalmanLog["x"].append(kalmanLogPrev["x"][boxIdx])
                    kalmanLog["P"].append(kalmanLogPrev["P"][boxIdx])
                    kalmanLog["Z"].append(np.empty((0, 0)))
                    kalmanLog["time"].append(kalmanLogPrev["time"][boxIdx] + 1)
        else:
            alpha = [0] * len(frameBox)

            kalmanLog["track id"] = list(frameBox.id)
            kalmanLog["x"] = [x0] * len(frameBox)
            kalmanLog["P"] = [P0ped if currentType == "Pedestrian" else P0car if currentType == "Car" else P0cyc for currentType in types]
            kalmanLog["Z"] = list(np.array([boxLeft, boxTop, boxRight, boxBottom]).T)
            kalmanLog["time"] = [0] * len(frameBox)

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
            kalmanLog["x"][idx] = x
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
    seq = 3
    cam = "left"

    dataPath = currentPath + f"34759_final_project_rect/seq_0{seq}/image_0{2 if cam.lower() == 'left' else 3}/"
    dataPNG = dataPath + "data/"
    dataTime = dataPath + "timestamps.txt"

    data = glob.glob(f"{dataPNG}*.png")
    timeStamp = open(dataTime)
    timeData = timeStamp.readlines()
    timeStamp.close()
    time1 = datetime.datetime.strptime(timeData[0][:-4], "%Y-%m-%d %H:%M:%S.%f")
    time2 = datetime.datetime.strptime(timeData[1][:-4], "%Y-%m-%d %H:%M:%S.%f")
    deltaT = (time2 - time1).total_seconds()

    states = 4

    x0 = np.zeros((3 * states, 1))
    u = np.zeros(x0.shape)
    P0car = np.diag([100000, 100, 0.1] * states)
    P0ped = np.diag([100000, 0.01, 0.000001] * states)
    P0cyc = np.diag([100000, 0.01, 0.0001] * states)
    R = 0.0001 * np.eye(states)

    F = np.eye(x0.shape[0])
    for i in range(F.shape[0]):
        try:
            F[i, i + 1] = deltaT if i % 3 != 2 and i < F.shape[0] - 1 else 0
            F[i, i + 2] = 0.5 * deltaT ** 2 if i % 3 == 0 and i < F.shape[0] - 2 else 0
        except:
            pass

    H = np.zeros((R.shape[0], x0.shape[0]))
    for i in range(H.shape[0]):
        H[i, i * 3] = 1
    
    kalman0 = {"x": x0, "u": u, "Pcar": P0car, "Pped": P0ped, "Pcyc": P0cyc, "F": F, "H": H, "R": R}

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
    save(h, w, 1 / deltaT, f"Results/ResultSeq{seq}{cam}")