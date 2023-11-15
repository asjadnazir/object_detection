import train_model
import os
import cv2
from ultralytics import YOLO


if not os.path.exists('runs/detect/train'):
    train_model.train()

cap = cv2.VideoCapture(1)

classNames = []

with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

model_path = os.path.join('.', 'models', 'airpods_detector.pt')
airpods_detector = YOLO(model_path)
threshold = 0.5
class_name_dict = {0: 'Airpods'}

while True:
    success, frame = cap.read()

    # yolov8 basic classes
    classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
    print(classIds, bbox)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds, confs.flatten(), bbox):
            if int(classId) != 74:
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0), 2)

    # Airpods
    airpods = airpods_detector(frame)[0]
    for airpod in airpods.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = airpod
            if score > 0.3:
                print('Airpods found ' + str(class_id) + ' ' + str(score))
                box = (int(x1), int(y1), int(x2), int(y2))  # Create a box tuple
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 50, 255),
                              thickness=2)  # Fix the rectangle coordinates
                cv2.putText(frame, 'Airpods', (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 50, 255), 2)

                # cv2.imshow("Found", frame)
                # cv2.waitKey(0)

    cv2.imshow("Output", frame)
    cv2.waitKey(1)


