from ultralytics import YOLO ; import cv2 ; import cvzone ; import math ; from sort import Sort
import numpy as np

cap = cv2.VideoCapture("0") # 0 for webcam and mp4 path for video
cv2.namedWindow("frame" , cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame" , cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)

model_1 = YOLO("Models/PlateAndCar.pt" , verbose = False)
model_2 = YOLO("Models/NumAndChars.pt" , verbose = False)
ClassNames_1 = model_1.names
ClassNames_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'd', 'ein', 'g', 'gh', 'h', 'l', 'm', 'malul', 'n', 's', 'sad', 't', 'ta', 'taxi', 'v', 'y']

Tracker = Sort(max_age = 60 , min_hits = 3 , iou_threshold = 0.3)
Tracked = []

while True:
    ret , frame = cap.read()
    results = model_1.predict(frame , stream = True)
    Detections = np.empty((0 , 5))

    for result in results:
            boxes = result.boxes
    for box in boxes:
        x1 , y1 , x2 , y2 = box.xyxy[0]
        x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
        conf = math.ceil(box.conf[0]*100)/100
        cls = int(box.cls[0])
        if ClassNames_1[cls] == "vehicle":
            if conf >= 0.7:
                TrackerBox = np.array([x1 , y1 , x2 , y2 , conf])
                Detections = np.vstack((Detections , TrackerBox))

                crop = frame[y1:y2 , x1:x2]
                results = model_1.predict(crop , stream = True)
                for result in results:
                    boxes = result.boxes
                for box in boxes:
                    plate_x1 , plate_y1 , plate_x2 , plate_y2 = box.xyxy[0]
                    plate_x1 , plate_y1 , plate_x2 , plate_y2 = int(plate_x1) , int(plate_y1) , int(plate_x2) , int(plate_y2)
                    conf = math.ceil(box.conf[0]*100)/100
                    cls = int(box.cls[0])
                    if ClassNames_1[cls] != "vehicle":
                        if conf >= 0.7:
                            cvzone.cornerRect(crop , (plate_x1 , plate_y1 , plate_x2-plate_x1 , plate_y2-plate_y1) , colorR = (0 , 0 , 255) , colorC = (0 , 0 , 255) , t = 2 , rt = 2)

                            new_crop = crop[plate_y1:plate_y2 , plate_x1:plate_x2] 
    updates = Tracker.update(Detections)
    for update in updates:
        x1 , y1 , x2 , y2 , ID = update
        x1 , y1 , x2 , y2 , ID = int(x1) , int(y1) , int(x2) , int(y2) , int(ID)
        if Tracked.count(ID) == 0:
            Tracked.append(ID)
            results = model_2.predict(new_crop , stream = True)
            names = {}
            show = []
            for result in results:
                boxes = result.boxes
            for box in boxes:
                x1 , y1 , x2 , y2 = box.xyxy[0]
                x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
                conf_2 = math.ceil(box.conf[0]*100)/100
                cls_2 = int(box.cls[0])
                names.setdefault(x1 , ClassNames_2[cls_2])
                cv2.imshow("Plate" , new_crop)
                if cv2.waitKey(1) == ord("g"):
                    pass
            for key , value in sorted(names.items()):
                show.append(value)

    cvzone.putTextRect(frame , f"Plate:{"".join(show)}" , (100 , 300) , thickness = 8 , scale = 8 , colorR = (255 , 255 , 255) , colorT = (0 , 0 , 0))


    cv2.imshow("frame" , frame)
    if cv2.waitKey(0) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()