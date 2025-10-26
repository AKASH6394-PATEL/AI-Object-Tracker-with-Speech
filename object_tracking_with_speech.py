import cv2
import numpy as np
from norfair import Detection, Tracker
import pyttsx3
import threading
import queue

# -------- SPEECH QUEUE SETUP --------
speech_queue = queue.Queue()
# ------------------------------------

# -------- SPEECH WORKER FUNCTION --------
def speech_worker():
    print("Speech worker thread shuru ho gaya hai.")
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    
    while True:
        try:
            text_to_speak = speech_queue.get()
            if text_to_speak is None:
                break
            print(f"SPEAKING (Thread): {text_to_speak}")
            engine.say(text_to_speak)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech thread mein error: {e}")
            
    engine.stop()
    print("Speech worker thread band ho gaya.")
# -----------------------------------------------------------------

# -------- SPEECH THREAD KO SHURU KARO --------
threading.Thread(target=speech_worker, daemon=True).start()
# -------------------------------------------

# -------- NORFAIR TRACKER SETUP --------
tracker = Tracker(
    distance_function="mean_euclidean",
    distance_threshold=50,
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

# -------- NAYI SMART MEMORY --------
# Humne 'set' ki jagah 'dictionary' (dict) banayi hai
# Yeh ID ke saath label (naam) bhi store karega
detected_objects_memory = {}
# -----------------------------------

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        norfair_detections = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.1: # Confidence 10% rakha hai
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    if class_id < len(classes):
                        label = str(classes[class_id])
                        norfair_detection = Detection(points=np.array([[x1, y1], [x2, y2]]), data=label)
                        norfair_detections.append(norfair_detection)

        tracked_objects = tracker.update(detections=norfair_detections)

        for obj in tracked_objects:
            x1, y1 = obj.estimate[0]
            x2, y2 = obj.estimate[1]
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            
            obj_id = obj.id
            label = obj.last_detection.data
            
            # -------- YEHI HAI BADLAAV (SMART CHECK) --------
            # Check karo ki kya yeh ID pehli baar dikhi hai
            # YA PHIR, kya is ID ka label (naam) badal gaya hai
            if obj_id not in detected_objects_memory or detected_objects_memory[obj_id] != label:
                
                detected_objects_memory[obj_id] = label # Memory ko update karo
                
                speech_text = f"{label} detected"
                speech_queue.put(speech_text) # Queue mein bolne ke liye daal do
            # ---------------------------------------------
                
            color = (0, 255, 0) # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Object Tracking Feed (with Speech)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    speech_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking band kar diya.")