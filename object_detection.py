import cv2
import numpy as np

# -------- AI MODEL LOAD KARNA (YOLO) --------
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# -----------------------------------------------

# -------- WEBCAM SETUP --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Webcam nahi chal raha hai.")
    exit()
print("Webcam open ho gaya hai. 'q' dabaakar band karein...")
# ----------------------------------------------------

while True:
    # 1. Frame padho
    ret, frame = cap.read()
    if not ret:
        break
    height, width, channels = frame.shape

    # -------- 2. FRAME KO AI MODEL KE LIYE TAIYAAR KARNA --------
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # -------------------------------------------------------------

    class_ids = []
    confidences = []
    boxes = []

    # -------- 3. AI KE RESULTS KO SCREEN PAR DIKHANA --------
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # ---- YEHI HAI BADLAAV #1 ----
            # Sirf unhi detections ko lo jinka confidence 20% se zyada hai
            if confidence > 0.2:  # Pehle yeh 0.5 tha
                # Object ka center aur size (width, height) pata karo
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Box ke kone (coordinates) pata karo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # ---- YEHI HAI BADLAAV #2 ----
    # Yahaan bhi confidence 0.5 ki jagah 0.2 kar diya
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN # Text ka style

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]) # Object ka naam
            confidence_str = str(round(confidences[i] * 100, 2)) + "%" # Confidence %
            color = (0, 255, 0) # Green color

            # ---- BOX AUR TEXT BANANA ----
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + confidence_str, (x, y - 5), font, 1.5, color, 2)

    # 4. Frame ko screen par dikhao
    cv2.imshow("Object Detection Feed", frame)

    # 5. Quit key 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# -----------------------------------------------------------

# Sab kuch band kar do
cap.release()
cv2.destroyAllWindows()
print("Detection band kar diya.")