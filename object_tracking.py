import cv2
import numpy as np
from norfair import Detection, Tracker

# -------- NORFAIR TRACKER SETUP --------
#
# ---- YEHI HAI BADLAAV ----
# 'euclidean' ko badal kar 'mean_euclidean' kar diya hai
tracker = Tracker(
    distance_function="mean_euclidean",  # Pehle yeh "euclidean" tha
    distance_threshold=30,
)
# ----------------------------------------

# -------- AI MODEL LOAD KARNA (YOLO - Puraana Code) --------
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# -----------------------------------------------

# -------- WEBCAM SETUP (Puraana Code) --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam nahi chal raha hai.")
    exit()
print("Webcam open ho gaya hai. 'q' dabaakar band karein...")
# ----------------------------------------------------

while True:
    # 1. Frame padho (Puraana Code)
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    # -------- 2. DETECTION (Puraana Code) --------
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Norfair ke liye ek khaali detection list banao
    norfair_detections = []

    # AI ke results ko check karo (Puraana Code)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Hum sirf 'person' (ID 0) ko track karenge
            if class_id == 0 and confidence > 0.3: # Sirf 'person' ko 30% se zyada par
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # -------- 3. NORFAIR KO DETECTION DENA --------
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                norfair_detection = Detection(points=np.array([[x1, y1], [x2, y2]]))
                norfair_detections.append(norfair_detection)
                # --------------------------------------------------

    # -------- 4. TRACKER KO UPDATE KARNA --------
    tracked_objects = tracker.update(detections=norfair_detections)
    # --------------------------------------------------

    # -------- 5. TRACKING RESULTS KO DRAW KARNA --------
    for obj in tracked_objects:
        x1, y1 = obj.estimate[0]
        x2, y2 = obj.estimate[1]
        x, y = int(x1), int(y1)
        w, h = int(x2 - x1), int(y2 - y1)
        
        obj_id = obj.id
        
        color = (0, 255, 0) # Green
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"Person {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # ----------------------------------------------------

    # Frame ko screen par dikhao (Puraana Code)
    cv2.imshow("Object Tracking Feed", frame)

    # Quit key 'q' (Puraana Code)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# -----------------------------------------------------------

# Sab kuch band kar do (Puraana Code)
cap.release()
cv2.destroyAllWindows()
print("Tracking band kar diya.")