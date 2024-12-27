import cv2 ; from ultralytics import YOLO ; import math ; import cvzone

im = cv2.imread("Image/im1.png")
# cv2.namedWindow("frame" , cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("frame" , cv2.WND_PROP_FULLSCREEN , cv2.WINDOW_FULLSCREEN)

model = YOLO("../Models/NumAndChars.pt" , verbose = False)
lst = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'd', 'ein', 'g', 'gh', 'h', 'l', 'm', 'malul', 'n', 's', 'sad', 't', 'ta', 'taxi', 'v', 'y']

while True:
    results = model.predict(im , stream = True)

    for res in results:
        boxes = res.boxes
    for box in boxes:
        x1 , y1 , x2 , y2 = box.xyxy[0]
        x1 , y1 , x2 , y2 = int(x1) , int(y1) , int(x2) , int(y2)
        conf = math.ceil(box.conf[0]*100)/100
        cls = int(box.cls[0])
        cvzone.putTextRect(im , f"{lst[cls]}" , (x1 , y1-10) , scale = 1 , thickness = 1 , colorR = (255 , 255 , 255) , colorT = (0 , 0 , 0) , colorB = (255 , 255 , 255))
    cv2.imshow("frame" , im)
    if cv2.waitKey(0) == ord("q"):
        break

cv2.destroyAllWindows()