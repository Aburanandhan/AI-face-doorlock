import cv2

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

print("Opened:", cap.isOpened())

# Try automatic exposure
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Increase brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)

# Increase gain
cap.set(cv2.CAP_PROP_GAIN, 20)

print("Exposure:", cap.get(cv2.CAP_PROP_EXPOSURE))
print("Brightness:", cap.get(cv2.CAP_PROP_BRIGHTNESS))
print("Gain:", cap.get(cv2.CAP_PROP_GAIN))

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Exposure Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()