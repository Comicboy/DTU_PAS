from ultralytics import YOLO
import numpy as np
import cv2

def detect_objects(img):
    model = YOLO("kitti_yolo_runs/exp1/weights/best.pt")
    class_names = {0: 'Car', 1: 'Car', 2: 'Car', 3: 'Pedestrian', 4: 'Pedestrian', 5: 'Cyclist', 6: 'Car', 7: 'Other'}

    results = model(img)

    for i, result in enumerate(results):
        try:
            boxes = result.boxes
        except Exception as e:
            print(f"Error processing result {i}: {e}")
            continue
        
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
            x1, y1, x2, y2 = map(int, coords)
            cls = int(box.cls[0].item()) if hasattr(box.cls[0], 'item') else int(box.cls[0])

            color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, class_names[cls], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img, results