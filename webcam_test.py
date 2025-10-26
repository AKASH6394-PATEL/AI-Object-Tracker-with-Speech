import cv2
import numpy as np

# Webcam ko capture karna shuru karo (0 ka matlab hai default webcam)
cap = cv2.VideoCapture(0)

# Check karo ki webcam sahi se open hua ya nahi
if not cap.isOpened():
    print("Error: Webcam nahi chal raha hai. Please check karein.")
    exit()

print("Webcam successfully open ho gaya hai.")
print("Window se bahar aane ke liye 'q' key dabaayein...")

# Ek 'while' loop jo tab tak chalega jab tak hum use band na karein
while True:
    # 1. Webcam se ek-ek frame (photo) padho
    ret, frame = cap.read()

    # Agar frame sahi se nahi padha (ret is False), toh loop band kar do
    if not ret:
        print("Error: Frame nahi mil raha hai.")
        break

    # 2. Frame ko screen par dikhao
    # 'Webcam Feed' uss window ka naam hai jo khulegi
    cv2.imshow('Webcam Feed', frame)

    # 3. Quit (Bahar nikalne) ka intezaar karo
    # Yeh 1 millisecond wait karega ki koi key press hui ya nahi
    # Agar 'q' key press hoti hai, toh loop ko 'break' (tod) do
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Loop khatam hone ke baad, sab kuch band kar do
print("Webcam band kar rahe hain...")
cap.release()          # Webcam ko chhod do
cv2.destroyAllWindows() # Saari windows band kar do