import sys
sys.path.append('/path/to/sort')

import cv2
from ultralytics import YOLO
import numpy as np
import os
from sort import Sort

model = YOLO("yolov9c.pt")
dir_path = '/Users/misaalsingh/Documents/tensorflow-test1/video'
sort_tracker = Sort()

def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    results = predict(chosen_model, img, classes, conf)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])
            detections.append([x1, y1, x2, y2, box.conf[0]])
    detections = np.array(detections)
    tracked_objects = sort_tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), int(obj[4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    return img, tracked_objects

def calculate_speed(tracked_objects, fps, pixel_to_meter_ratio):
    speeds = {}
    for obj in tracked_objects:
        obj_id = int(obj[4])
        if obj_id not in speeds:
            speeds[obj_id] = {'positions': [], 'speeds': []}
        center_x = (obj[0] + obj[2]) / 2
        center_y = (obj[1] + obj[3]) / 2
        speeds[obj_id]['positions'].append((center_x, center_y))
        if len(speeds[obj_id]['positions']) > 1:
            dx = speeds[obj_id]['positions'][-1][0] - speeds[obj_id]['positions'][-2][0]
            dy = speeds[obj_id]['positions'][-1][1] - speeds[obj_id]['positions'][-2][1]
            distance = np.sqrt(dx ** 2 + dy ** 2) * pixel_to_meter_ratio
            speed = distance * fps * 3.6  # Convert m/s to km/h
            speeds[obj_id]['speeds'].append(speed)
    return speeds

def pred_video(video, fps=30, pixel_to_meter_ratio=0.05):  # Assume 0.05 meters per pixel for example
    v_path = os.path.join(dir_path, video)
    cap = cv2.VideoCapture(v_path)
    while True:
        success, img = cap.read()
        if not success:
            break
        result_img, tracked_objects = predict_and_detect(model, img, classes=[], conf=0.5)
        speeds = calculate_speed(tracked_objects, fps, pixel_to_meter_ratio)
        for obj_id, data in speeds.items():
            if data['speeds']:
                avg_speed = np.mean(data['speeds'])
                cv2.putText(result_img, f"Speed: {avg_speed:.2f} km/h", (int(tracked_objects[0][0]), int(tracked_objects[0][1]) - 30),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)
        cv2.imshow("Video", result_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

test_video = np.random.choice(df["Videos"].values.tolist())
pred_video(test_video)
